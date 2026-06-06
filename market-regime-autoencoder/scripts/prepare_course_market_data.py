from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the course market_data.xlsx file into a clean feature CSV."
    )
    parser.add_argument("--input", required=True, help="Path to market_data.xlsx.")
    parser.add_argument("--sheet", default="US", help="Excel sheet name.")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["_TY", "ED", "_MKT"],
        help="Columns to use as features.",
    )
    parser.add_argument(
        "--frequency",
        choices=["W", "M", "Q"],
        default="M",
        help="Output frequency: weekly, monthly, or quarterly.",
    )
    parser.add_argument(
        "--transform",
        choices=["pct_change", "diff", "level"],
        default="pct_change",
        help="Feature transformation after frequency alignment.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/course_market_features.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_excel(args.input, sheet_name=args.sheet, engine="openpyxl")
    if "Date" not in df.columns:
        raise ValueError("Expected a Date column in the Excel sheet.")

    missing = [col for col in args.columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing requested columns: {missing}")

    df = df[["Date", *args.columns]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    if args.frequency in {"M", "Q"}:
        resample_rule = {"M": "ME", "Q": "QE"}[args.frequency]
        df = df.resample(resample_rule).last()

    if args.transform == "pct_change":
        df = df.pct_change()
    elif args.transform == "diff":
        df = df.diff()

    df = df.dropna().reset_index().rename(columns={"Date": "date"})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")
    print(f"Frequency: {args.frequency}, transform: {args.transform}")
    print(f"Columns: {', '.join(args.columns)}")


if __name__ == "__main__":
    main()
