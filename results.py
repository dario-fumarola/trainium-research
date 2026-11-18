"""Read the archived aggregate scores and calculate the headline comparisons."""
import csv
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent / "evidence"


def score_table(filename):
    with (EVIDENCE / filename).open(newline="") as stream:
        return {(row["cohort"], row["model"]): float(row["full_window_NLL"])
                for row in csv.DictReader(stream)}


def report(filename):
    return json.loads((EVIDENCE / filename).read_text())


def main():
    scaffold = score_table("scaffold.csv")
    mixture = score_table("source-mixture.csv")
    print("Qwen3-1.7B continued training | gain in nats/token (positive favors intervention)")
    print("order    trace vs raw    mixed vs base, math    mixed vs base, English")
    for order in (11, 12):
        raw = scaffold["CONFIRM", f"scaffold_raw-seed{order}"]
        trace = scaffold["CONFIRM", f"scaffold_trace-seed{order}"]
        math_gain, english_gain = (
            mixture[cohort, "B0"] - mixture[cohort, f"mixed-seed{order}"]
            for cohort in ("MATH_CONFIRM", "ENGLISH_CONFIRM")
        )
        print(f"{order:5} {raw - trace:+15.6f} {math_gain:+22.6f} {english_gain:+25.6f}")

    closure = report("pause-closure.json")
    hours = sum(attempt["assigned_chip_hours"] for attempt in closure["attempts"])
    print(f"\nPause pilot | {len(closure['attempts'])} attempts, {hours:.3f} assigned chip-hours")
    print(f"Hardware qualification passed: {closure['hardware_qualification_passed']}")

    print("\nFrom scratch | ordinary NLL minus latent NLL (positive favors latent)")
    print("comparison       broad text    arithmetic answers")
    for label, prefix in (("Equal data", "equal-data"), ("Matched work", "matched-work-recovery-v1")):
        broad = report(f"{prefix}-broad.json")["comparisons"]["ordinary"]["scores"]
        math = report(f"{prefix}-arithmetic.json")["means"]
        broad_gain = broad["control"]["broad_nll"] - broad["candidate"]["broad_nll"]
        math_gain = math["ordinary"]["answer_nll"]["mean"] - math["latent"]["answer_nll"]["mean"]
        print(f"{label:14} {broad_gain:+12.6f} {math_gain:+21.6f}")


if __name__ == "__main__":
    main()
