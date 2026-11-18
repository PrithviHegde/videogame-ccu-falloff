# Contributing to Steam Player Observatory

This is how we work on the project together. You don't need to be an experienced coder to contribute.

## The one rule

**Never commit directly to `main`.** Every change to `main` is published to the live app within a minute or two. All changes go through a pull request (PR) that one teammate has looked at.

## Making a change

1. **Get the latest code.**
   ```bash
   git checkout main
   git pull
   ```
2. **Create a branch** named after what you're doing.
   ```bash
   git checkout -b falloff-marker
   ```
3. **Make your change and try it locally** (see "Running the app" below).
4. **Commit and push your branch.**
   ```bash
   git add .
   git commit -m "Add falloff point marker to daily chart"
   git push -u origin falloff-marker
   ```
5. **Open a pull request** on GitHub. Say what you changed and why. Add a screenshot if the app looks different.
6. **Wait for the green check.** GitHub runs the tests automatically on every PR. If the check is red, click it to see what failed, or ask in the team chat.
7. **Ask one teammate to review**, then merge once they approve.

Keep PRs small: one feature or one fix each. Small PRs get reviewed faster and rarely conflict with someone else's work.

## Running the app

First time only:

```bash
python -m venv .venv
# Windows:    .venv\Scripts\activate
# Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt pytest
```

After that, activate the environment (the second line above) and run:

```bash
streamlit run app.py
```

The app opens at http://localhost:8501 and reloads when you save a file.

## Running the tests

```bash
pytest
```

It takes a few seconds. There are two kinds of test in `tests/`:

- `test_smoke.py` opens the real app on both views and checks that nothing crashes.
- `test_data.py` checks the calculations (cleaning, filtering, statistics, ranking) on small made-up tables.

If you add a new calculation, please add a small test for it. If you only change the layout or wording, the existing tests are enough.

## Contributing without writing code

- **Report a problem or suggest an idea**: open an Issue on GitHub. Describe what you saw, what you expected, and add a screenshot if you can.
- **Give design feedback**: comment on an Issue or a PR. Screenshots with notes drawn on them are welcome.
- **Fix wording or a typo**: open the file on GitHub, click the pencil icon, make the edit, and choose "Create a new branch for this commit and start a pull request". GitHub handles the rest.

## Data rules

- Don't commit raw data (`.csv`, `.zip`). Only the compact `.parquet` files in `processed/` belong in the repo. The `.gitignore` already blocks the rest.
- Don't use data from SteamDB; its policies prohibit scraping. Steam's official Web API and properly licensed datasets are fine.
- CCU means players online at the same time. It is not unique players or retention, so don't label it that way in the app.
