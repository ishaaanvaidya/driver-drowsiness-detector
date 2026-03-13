"""Drive session summary report generator"""
import pandas as pd
from pathlib import Path


ALERT_LEVELS = ['NORMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
DROWSY_LEVELS = {'HIGH', 'CRITICAL'}


def generate_summary(csv_path):
    """Read a session CSV and print/save a formatted drive summary.

    Computes:
    - Total session duration in minutes
    - Percentage of time at each alert level
    - Estimated time spent drowsy (HIGH + CRITICAL)
    - Number of microsleep episodes (consecutive CRITICAL frames > 15)
    - Average EAR and average blink_rate
    - Drowsiness score out of 100 (100 = fully alert, 0 = severely drowsy)

    Prints a formatted report and saves it as a .txt file next to the CSV.

    Args:
        csv_path: str or Path to the session CSV file

    Returns:
        Path to the saved .txt summary file, or None if the CSV is empty
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if df.empty:
        print('No data recorded in this session.')
        return None

    total_frames = len(df)

    # Total duration (seconds between first and last timestamp)
    duration_seconds = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
    duration_minutes = max(duration_seconds / 60.0, 0.0)

    # Alert level percentages
    level_counts = df['alert_level'].value_counts()
    level_pct = {
        lvl: (level_counts.get(lvl, 0) / total_frames) * 100
        for lvl in ALERT_LEVELS
    }

    # Drowsy time (HIGH + CRITICAL)
    drowsy_frames = df[df['alert_level'].isin(DROWSY_LEVELS)].shape[0]
    drowsy_pct = (drowsy_frames / total_frames) * 100

    # Microsleep episodes: runs of consecutive CRITICAL frames longer than 15
    microsleep_count = 0
    streak = 0
    for flag in df['microsleep_flag']:
        if flag == 1:
            streak += 1
        else:
            if streak > 15:
                microsleep_count += 1
            streak = 0
    if streak > 15:
        microsleep_count += 1

    # Average metrics
    avg_ear = df['ear'].mean()
    avg_blink_rate = df['blink_rate'].mean()

    # Drowsiness score: 100 = fully alert, 0 = severely drowsy
    score = round(100 * (1 - drowsy_frames / total_frames))

    # Peak alert level reached
    peak = 'NORMAL'
    for lvl in ALERT_LEVELS:
        if lvl in level_counts.index:
            peak = lvl

    # Emoji rating and recommendation
    if score >= 80:
        rating = '🟢'
        recommendation = 'Great drive! You stayed alert throughout.'
    elif score >= 50:
        rating = '🟡'
        recommendation = 'Take a short break at the next stop to refresh yourself.'
    else:
        rating = '🔴'
        recommendation = '⚠️  Please stop driving and rest immediately.'

    lines = [
        '=' * 52,
        '🚗  Drive Summary',
        '=' * 52,
        f'  Total Drive Time    : {duration_minutes:.1f} min',
        f'  Drowsiness Score    : {score}/100  {rating}',
        f'  Peak Alert Level    : {peak}',
        f'  Microsleep Episodes : {microsleep_count}',
        '',
        '  Time at Each Alert Level:',
    ]
    for lvl in ALERT_LEVELS:
        lines.append(f'    {lvl:<10}: {level_pct[lvl]:5.1f}%')
    lines += [
        '',
        f'  Time Spent Drowsy   : {drowsy_pct:.1f}%',
        f'  Average EAR         : {avg_ear:.3f}',
        f'  Average Blink Rate  : {avg_blink_rate:.1f} blinks/min',
        '',
        f'  💡 {recommendation}',
        '=' * 52,
    ]

    report = '\n'.join(lines)
    print('\n' + report)

    # Save .txt file next to the CSV
    txt_path = csv_path.with_suffix('.txt')
    txt_path.write_text(report)

    return txt_path
