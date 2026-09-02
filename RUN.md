# Running the bot

## 1. Paper (start here — no keys, no risk)

```bash
cd ~/crypto-backtest/research
python3 bot.py            # loops, polls every 45s, decides on each closed 4h candle
```

- Trades log to `paper_trades.jsonl` (always) and account state to `bot_state.json`.
- `python3 journal.py` — summary: win rate, per-symbol P&L, vs backtest expectation.
- `python3 explain_trade.py XLMUSDT 8` — full annotated breakdown of recent trades.
- `python3 replay.py` — re-runs the bot's code over history and checks it against
  the backtester (they land within ~1 pt of return over 400d).
- Leave it a few weeks. The strategy makes ~1 trade every 2-3 days per symbol, and
  the edge only shows over months. If paper roughly tracks the backtest
  (~40% win rate, target exits > stop exits in $), the wiring is sound.

### Log to the waddle-ops dashboard (recommended)

1. In the **notes** Supabase project's SQL editor, run
   `supabase_paper_trades.sql` (creates `waddle_paper_trades`).
2. Run the bot with two extra env vars — same project URL, same service-role key
   the dashboard uses:

```bash
export WADDLE_SUPABASE_URL="https://<project>.supabase.co"
export WADDLE_SUPABASE_KEY="<service-role key>"
python3 bot.py
```

Every open/close is POSTed to Supabase (best-effort — a network failure never
stops trading). The **Bot** tab at waddle-ops shows the equity curve, win rate,
per-symbol P&L, open positions and recent-trade table, auto-refreshing each minute.

### Keep it alive on this Mac

```bash
# quick: survives terminal close, logs to file
nohup python3 bot.py > bot.log 2>&1 &

# proper: auto-restart on crash / reboot — install the launch agent
cp com.waddle.donchianbot.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.waddle.donchianbot.plist
launchctl start com.waddle.donchianbot
# stop:  launchctl unload ~/Library/LaunchAgents/com.waddle.donchianbot.plist
```

### Or run it on Railway (always-on, off your laptop)

This folder is a self-contained git repo with `requirements.txt`, `Procfile`,
`nixpacks.toml` and `railway.json` — Railway needs nothing else.

**First, the trade log has to go to Supabase** (Railway wipes its disk on every
deploy — state is restored from `waddle_bot_state`, see below). Run
`supabase_paper_trades.sql` in the notes Supabase project.

**Push the repo:**
```bash
cd ~/crypto-backtest/research
gh repo create waddle-donchian-bot --private --source=. --remote=origin --push
```

**On railway.app:** New Project → Deploy from GitHub → pick `waddle-donchian-bot`.
It auto-detects the config. Then in the service's **Variables** tab add:

| var | value |
|---|---|
| `WADDLE_SUPABASE_URL` | `https://<project>.supabase.co` |
| `WADDLE_SUPABASE_KEY` | the notes project's service-role key |

Deploy. Logs tab shows the same `equity $…` heartbeat every 45s. State survives
redeploys via Supabase; every trade lands in `waddle_paper_trades` for the
dashboard. Railway needs a paid/hobby plan (~$5/mo) to run 24/7.

## 2. Live on Bybit — testnet first

1. Create a Bybit **testnet** account (testnet.bybit.com), fund it with faucet USDT.
2. API key: **Contract → Orders & Positions only**. NOT withdrawals.
3. Export keys and run against testnet:

```bash
export BYBIT_API_KEY=...        # testnet key
export BYBIT_API_SECRET=...
export BYBIT_LIVE=1
python3 bot.py --live --testnet
```

Watch it place real testnet orders with attached stop-loss / take-profit. Confirm
`journal`-style behaviour matches paper for a week or two.

## 3. Live on mainnet — real $50

Same as above but a mainnet key and drop `--testnet`:

```bash
export BYBIT_API_KEY=...        # mainnet key, Orders & Positions only
export BYBIT_API_SECRET=...
export BYBIT_LIVE=1             # required guard for mainnet
python3 bot.py --live
```

- Start at `RISK_PCT = 0.005` (0.5%/leg) in `bot.py`. Only raise it after months
  of live data you trust.
- Entries go in as Market orders **with exchange-native stopLoss/takeProfit**, so
  a bot crash can't leave a naked position.
- On restart the bot reads live positions from Bybit — it does not assume its
  local state is correct.

## Dials (top of `bot.py`)

| const | meaning | sane range |
|---|---|---|
| `RISK_PCT` | equity risked to the stop per leg per trade | 0.005–0.01 |
| `MAX_LEVERAGE` | safety cap only; strategy uses ~0.5–1x in practice | 5 |
| `CFG['rr']` | reward:risk — 2.0 is the tested optimum | 2.0 |
| `LEGS` | symbols traded | XLM, XRP (+TRX/ADA optional) |
