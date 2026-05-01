# Driver Drowsiness Detector

Run the detector:

```powershell
uv run python main.py
```

Run the session dashboard:

```powershell
uv run streamlit run dashboard.py
```

The dashboard reads CSV logs from `data/sessions` and shows risk trends, eye behavior,
head pose, alert episodes, and run-to-run comparisons.
