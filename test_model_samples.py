#!/usr/bin/env python3
"""Test the trained model with diverse samples."""

import subprocess
import json

# Test samples with expected labels
test_samples = [
    # HUMAN SAMPLES - Informal/Social Media
    ("honestly idk why ppl are so pressed about this... like just let people live their lives??", "human", "reddit casual"),
    ("ngl this weather is absolutely trash today. cant wait for summer tbh", "human", "twitter informal"),
    ("LMAOOO did you see what happened at the game last night?? absolutely insane", "human", "social media reaction"),
    ("my cat just knocked over my coffee for the third time this week fml", "human", "personal anecdote"),

    # HUMAN SAMPLES - Formal/Academic
    ("The methodology employed in this study utilized a mixed-methods approach, combining quantitative analysis of survey data (n=342) with qualitative interviews.", "human", "academic paper"),
    ("Plaintiff hereby moves this Court for summary judgment pursuant to Federal Rule of Civil Procedure 56.", "human", "legal filing"),
    ("Our findings suggest a statistically significant relationship between variables X and Y (r=0.73, p<0.001).", "human", "research results"),

    # LLM SAMPLES - GPT-4 style
    ("I understand you're looking for advice on this topic. Let me provide you with a comprehensive overview of the key considerations involved.", "llm", "GPT-4 opening"),
    ("There are several important factors to consider when approaching this problem. First, we should examine the underlying assumptions. Second, we need to evaluate the available options.", "llm", "structured response"),
    ("That's a great question! The answer depends on several factors. Let me break this down for you step by step.", "llm", "enthusiastic helper"),

    # LLM SAMPLES - Claude style
    ("I appreciate your question about this complex topic. To provide a thorough response, I'll need to address multiple aspects of the issue.", "llm", "Claude opening"),
    ("While I understand your perspective, it's important to note that there are various viewpoints on this matter that merit consideration.", "llm", "balanced Claude"),

    # LLM SAMPLES - Generic/Instructional
    ("To begin, ensure all necessary materials are prepared. Next, carefully follow each step in sequence. Finally, verify the results match expectations.", "llm", "generic instructions"),
    ("This comprehensive guide will walk you through the essential steps needed to achieve your desired outcome effectively and efficiently.", "llm", "guide intro"),

    # EDGE CASES - Ambiguous
    ("The impact of social media on society is significant.", "?", "simple statement"),
    ("I think this is wrong.", "?", "short opinion"),
    ("Yes, that's correct.", "?", "brief agreement"),
    ("Error: Unable to process request.", "?", "error message"),
]

print("Testing LLM Detector v0.4.0")
print("=" * 80)
print()

correct_predictions = 0
total_predictions = 0

for text, expected, description in test_samples:
    # Escape quotes in text for shell command
    escaped_text = text.replace('"', '\\"')

    cmd = f'uv run llm-detector --model models/llm_detector_v0.4.0.joblib --text "{escaped_text}" --json'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            prediction = "llm" if data[0]["is_llm"] else "human"
            confidence = data[0]["confidence"] * 100

            # Check correctness (skip ambiguous cases)
            if expected != "?":
                is_correct = prediction == expected
                correct_predictions += is_correct
                total_predictions += 1
                symbol = "✓" if is_correct else "✗"
            else:
                symbol = "?"

            # Format output
            exp_str = expected.upper() if expected != "?" else "UNKN"
            pred_str = prediction.upper()

            print(f"{symbol} [{exp_str:5}] -> [{pred_str:5}] ({confidence:5.1f}%) | {description}")
            print(f"   '{text[:70]}{'...' if len(text) > 70 else ''}'")
            print()

        except json.JSONDecodeError:
            print(f"ERROR: Failed to parse JSON for: {description}")
            print(f"Output: {result.stdout}")
    else:
        print(f"ERROR: Command failed for: {description}")
        print(f"Error: {result.stderr}")

# Summary
print("=" * 80)
print(f"SUMMARY: {correct_predictions}/{total_predictions} correct predictions ({correct_predictions/total_predictions*100:.1f}% accuracy)")
print()

# Additional test with longer text
print("Testing with longer text samples...")
print("-" * 80)

longer_samples = [
    # Human blog post
    ("""So I've been thinking about this whole AI thing lately, and honestly? It's getting weird.
Like, I was talking to my coworker yesterday and she literally couldn't tell if an email was
written by ChatGPT or not. That's kinda scary when you think about it. But also, maybe we're
overthinking it? I mean, back in the day people probably freaked out about calculators too.
Still, something feels different this time. IDK, maybe I'm just getting old lol.""", "human", "blog post"),

    # LLM essay
    ("""The emergence of artificial intelligence represents a paradigm shift in how we approach
problem-solving and decision-making. This technological revolution brings both opportunities
and challenges that society must carefully navigate. On one hand, AI systems enhance productivity
and enable breakthrough discoveries. On the other hand, they raise important questions about
employment, privacy, and human agency. As we move forward, it is crucial that we develop
frameworks to ensure these powerful tools benefit humanity as a whole.""", "llm", "essay intro"),
]

for text, expected, description in longer_samples:
    escaped_text = text.replace('"', '\\"').replace('\n', ' ')

    cmd = f'uv run llm-detector --model models/llm_detector_v0.4.0.joblib --text "{escaped_text}" --json'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        data = json.loads(result.stdout)
        prediction = "llm" if data[0]["is_llm"] else "human"
        confidence = data[0]["confidence"] * 100
        sentence_count = data[0]["details"]["sentence_count"]

        is_correct = prediction == expected
        symbol = "✓" if is_correct else "✗"

        print(f"{symbol} [{expected.upper():5}] -> [{prediction.upper():5}] ({confidence:5.1f}%) | {description}")
        print(f"   Sentences analyzed: {sentence_count}")
        print()