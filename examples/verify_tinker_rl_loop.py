import os
import asyncio
import torch
import numpy as np
import random, re, math
import tinker
from tinker import types
from transformers import AutoTokenizer
import argparse
import logging
import json
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("tinker_rl_training.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_URL = "http://localhost:8001"
# Default model if not specified
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct" 

SYSTEM_PROMPT = """Add adjacent pairs of numbers, then multiply the results.

Example 1: 3, 5, 2, 4
- Add pairs: 3+5=8, 2+4=6
- Multiply: 8×6=48
- Answer: <answer>48</answer>

Example 2 (Two numbers): 7, 2
- Add pairs: 7+2=9
- Only one result, so product is 9
- Answer: <answer>9</answer>

Now solve the problem below. Put your final answer in <answer>X</answer> tags."""

def generate_problem():
    n = random.choice([2, 4]) # Smaller for testing
    nums = [random.randint(1, 9) for _ in range(n)]
    sums = [nums[i] + nums[i+1] for i in range(0, len(nums), 2)]
    answer = math.prod(sums)
    return nums, sums, answer

def compute_reward(response, correct_answer):
    answer_match = re.search(r'<answer>\s*(\d+)\s*</answer>', response)
    if answer_match:
        if int(answer_match.group(1)) == correct_answer:
            return 1.1
        return 0.1
    return 0.0

def compute_advantages(rewards):
    rewards = np.array(rewards)
    if rewards.std() < 1e-8:
        return [0.0] * len(rewards)
    return ((rewards - rewards.mean()) / (rewards.std() + 1e-8)).tolist()

import argparse

def make_rl_datum(prompt_tokens, completion_tokens, completion_logprobs, advantage):
    full_tokens = prompt_tokens + list(completion_tokens)
    input_tokens, target_tokens = full_tokens[:-1], full_tokens[1:]
    
    # We want advantages and logprobs to align with target_tokens
    # prompt length: n_prompt, completion length: n_completion
    # Since input_tokens cuts off the last completion token, target_tokens cuts off the first prompt token.
    # The action mask marks where our target_tokens belong to the completion.
    n_prompt_targets = len(prompt_tokens) - 1
    n_completion_targets = len(completion_tokens)
    
    # For action_mask, 0 for prompt targets, 1 for completion targets
    action_mask = [0.0] * n_prompt_targets + [1.0] * n_completion_targets
    
    # logprobs/advantages from the completion are matched 1:1 with completion targets
    loss_fn_inputs = {
        "target_tokens": target_tokens,
        "action_mask": types.TensorData(data=action_mask, dtype="float32", shape=[len(action_mask)]),
        "logprobs": [0.0] * n_prompt_targets + list(completion_logprobs),
        "advantages": [0.0] * n_prompt_targets + [advantage] * n_completion_targets
    }

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=loss_fn_inputs
    )

async def verify_lora_sampling_only(args):
    print(f"Connecting to Tinker Server at {BASE_URL}...")
    client = tinker.ServiceClient(base_url=BASE_URL, api_key="tml-dummy")
    
    print(f"Creating training client for {args.model}...")
    training_client = await client.create_lora_training_client_async(base_model=args.model, rank=8)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    sampling_client = training_client.save_weights_and_get_sampling_client()
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, 
                {"role": "user", "content": "1, 2, 3, 4"}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    print("Generating 5 LoRA samples for test prompt: 1, 2, 3, 4")
    samp_future = sampling_client.sample(
        prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
        num_samples=5,
        sampling_params=types.SamplingParams(max_tokens=256, temperature=0.8)
    )
    
    resp = samp_future.result()
    print("\n--- LoRA Samples ---")
    for i, seq in enumerate(resp.sequences):
        text = tokenizer.decode(seq.tokens)
        print(f"Sample {i+1}:")
        print(f"Tokens: {seq.tokens}")
        print(f"Text: {text}")
        print("-" * 20)
    print("\nLoRA Sampling Verification Complete!")

async def verify_pure_sampling_only(args):
    print(f"Connecting to Tinker Server at {BASE_URL}...")
    client = tinker.ServiceClient(base_url=BASE_URL, api_key="tml-dummy")
    
    print(f"Creating sampling client directly for {args.model}...")
    sampling_client = await client.create_sampling_client_async(base_model=args.model)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, 
                {"role": "user", "content": "1, 2, 3, 4"}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    print("Generating 5 pure samples for test prompt: 1, 2, 3, 4")
    samp_future = sampling_client.sample(
        prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
        num_samples=5,
        sampling_params=types.SamplingParams(max_tokens=256, temperature=0.8)
    )
    
    resp = samp_future.result()
    print("\n--- Pure Samples ---")
    for i, seq in enumerate(resp.sequences):
        text = tokenizer.decode(seq.tokens)
        print(f"Sample {i+1}:")
        print(f"Tokens: {seq.tokens}")
        print(f"Text: {text}")
        print("-" * 20)
    print("\nPure Sampling Verification Complete!")

async def main():
    parser = argparse.ArgumentParser(description="Verify Tinker RL Loop")
    parser.add_argument("--sample-only", choices=["pure", "lora"], help="Only verify the sampling endpoint (pure or lora)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of RL iterations to run")
    parser.add_argument("--num-problems", type=int, default=16, help="Number of distinct problems per iteration")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of rollouts per problem")
    parser.add_argument("--num-eval", type=int, default=20, help="Number of eval problems for holdout validation")
    parser.add_argument("--microbatch-size", type=int, default=4, help="Batch size for forward-backward passes to prevent OOM")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model to use")
    args = parser.parse_args()

    if args.sample_only == "pure":
        await verify_pure_sampling_only(args)
        return
    elif args.sample_only == "lora":
        await verify_lora_sampling_only(args)
        return
    
    logger.info(f"Connecting to Tinker Server at {BASE_URL}...")
    client = tinker.ServiceClient(base_url=BASE_URL, api_key="tml-dummy")
    
    logger.info(f"Creating training client for {args.model}...")
    training_client = await client.create_lora_training_client_async(base_model=args.model, rank=8)
    
    # We need a local tokenizer to decode/encode for reward logic
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    logger.info(f"Generating {args.num_eval} fixed eval problems for holdout validation...")
    eval_problems = [generate_problem() for _ in range(args.num_eval)]
    
    history = []

    for i in range(args.iterations):
        logger.info(f"\n--- Iteration {i+1}/{args.iterations} ---")
        
        # 1. Rollouts
        sampling_client = training_client.save_weights_and_get_sampling_client()
        
        problems = [generate_problem() for _ in range(args.num_problems)]
        all_rollouts = []
        
        logger.info("Generating rollouts...")
        for nums, sums, ans in problems:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, 
                        {"role": "user", "content": ", ".join(map(str, nums))}]
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            
            # Use async for batching!
            samp_future = sampling_client.sample(
                prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
                num_samples=args.num_samples,
                sampling_params=types.SamplingParams(max_tokens=256, temperature=0.8)
            )
            all_rollouts.append((prompt_tokens, ans, samp_future))
            
        # Wait for all samples
        training_rollouts = []
        rewards = []
        format_correct = 0
        math_correct = 0

        for prompt_tokens, ans, future in all_rollouts:
            resp = future.result()
            for seq in resp.sequences:
                text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                reward = compute_reward(text, ans)
                logger.info(f"Sample: {text!r} | Reward: {reward}")
                rewards.append(reward)
                if reward >= 0.1: format_correct += 1
                if reward >= 1.0: math_correct += 1
                training_rollouts.append({
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": seq.tokens,
                    "logprobs": seq.logprobs,
                    "reward": reward
                })
        
        mean_reward = np.mean(rewards)
        acc_format = format_correct / len(rewards)
        acc_math = math_correct / len(rewards)

        logger.info(f"Mean Reward: {mean_reward:.4f} | Format Acc: {acc_format:.0%} | Math Acc: {acc_math:.0%}")
        
        # 2. Train
        advantages = compute_advantages(rewards)
        datums = [
            make_rl_datum(r["prompt_tokens"], r["completion_tokens"], r["logprobs"], adv)
            for r, adv in zip(training_rollouts, advantages)
        ]
        
        logger.info(f"Submitting training step with {len(datums)} rollouts (microbatch size: {args.microbatch_size})...")
        fb_futures = []
        for j in range(0, len(datums), args.microbatch_size):
            chunk = datums[j:j + args.microbatch_size]
            fb_futures.append(await training_client.forward_backward_async(chunk, "importance_sampling"))
            
        opt_future = await training_client.optim_step_async(types.AdamParams(learning_rate=1e-4))
        
        losses = []
        for fb_fut in fb_futures:
            fb_res = fb_fut.result()
            losses.append(fb_res.metrics.get('loss:mean', 0.0))
            
        opt_res = opt_future.result()
        
        loss = np.mean(losses) if losses else 0.0
        logger.info(f"Loss: {loss:.4f}, LR: {opt_res.metrics['lr']}")

        # 3. Eval Pass (Greedy Decoding on Holdout Set)
        logger.info("Running holdout validation eval pass...")
        eval_rollouts = []
        for nums, sums, ans in eval_problems:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, 
                        {"role": "user", "content": ", ".join(map(str, nums))}]
            prompt_tokens_eval = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            
            # Temperature 0.0 for greedy decoding
            eval_future = sampling_client.sample(
                prompt=types.ModelInput.from_ints(tokens=prompt_tokens_eval),
                num_samples=1,
                sampling_params=types.SamplingParams(max_tokens=256, temperature=0.0)
            )
            eval_rollouts.append((ans, eval_future))

        eval_math_correct = 0
        eval_format_correct = 0
        
        for ans, future in eval_rollouts:
            resp = future.result()
            seq = resp.sequences[0]
            text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            reward = compute_reward(text, ans)
            if reward >= 0.1: eval_format_correct += 1
            if reward >= 1.0: eval_math_correct += 1
            
        eval_acc_format = eval_format_correct / len(eval_rollouts) if len(eval_rollouts) > 0 else 0
        eval_acc_math = eval_math_correct / len(eval_rollouts) if len(eval_rollouts) > 0 else 0
        
        logger.info(f"Eval Format Acc: {eval_acc_format:.0%} | Eval Math Acc: {eval_acc_math:.0%}")

        history.append({
            "iteration": i + 1,
            "reward": mean_reward,
            "format_acc": acc_format,
            "math_acc": acc_math,
            "eval_format_acc": eval_acc_format,
            "eval_math_acc": eval_acc_math,
            "loss": loss
        })

    logger.info("\nVerification Complete! Saving metrics and plotting...")
    
    with open("tinker_rl_metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plotting
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        iters = [h["iteration"] for h in history]
        
        axes[0].plot(iters, [h["reward"] for h in history], 'b-o')
        axes[0].set_title("Mean Reward")
        axes[0].set_xlabel("Iteration")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(iters, [h["format_acc"] for h in history], 'g--o', label="Train Format Acc")
        axes[1].plot(iters, [h["math_acc"] for h in history], 'g-s', label="Train Math Acc")
        axes[1].plot(iters, [h["eval_format_acc"] for h in history], 'm--^', label="Eval Format Acc")
        axes[1].plot(iters, [h["eval_math_acc"] for h in history], 'm-*', label="Eval Math Acc")
        axes[1].set_title("Accuracy")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_xlabel("Iteration")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(iters, [h["loss"] for h in history], 'r-o')
        axes[2].set_title("RL Loss")
        axes[2].set_xlabel("Iteration")
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("tinker_rl_learning_curve.png", dpi=150)
        logger.info("Saved plot to tinker_rl_learning_curve.png")
    except Exception as e:
        logger.info(f"Could not generate plot: {e}")

if __name__ == "__main__":
    asyncio.run(main())
