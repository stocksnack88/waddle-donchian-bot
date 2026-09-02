-- Trade log for the Donchian bot. Run once in the Supabase SQL editor of the
-- SAME project that holds waddle_notes (the "notes" project).
--
-- The bot writes here via the service-role key (env WADDLE_SUPABASE_KEY).
-- The waddle-ops dashboard reads here via the same key. RLS is on with no
-- policies, so anon / authenticated clients get nothing.

create table if not exists public.waddle_paper_trades (
  id            bigint generated always as identity primary key,
  event         text not null check (event in ('open', 'close')),
  mode          text not null default 'paper',      -- paper | testnet | live
  symbol        text not null,
  side          smallint,                           -- 1 long, -1 short
  qty           double precision,
  price         double precision,                   -- fill price (entry on open, exit on close)
  entry_price   double precision,                   -- original entry, on close rows
  sl            double precision,
  tp            double precision,
  pnl           double precision,                   -- realised, on close rows
  reason        text,                               -- stop | target | signal | flip, on close
  tag           text,                               -- entry reason, on open
  candle_ts     timestamptz,                        -- the 4h candle that triggered
  equity_after  double precision,                   -- account cash after this event
  created_at    timestamptz not null default now()
);

create index if not exists waddle_paper_trades_created_idx
  on public.waddle_paper_trades (created_at);

alter table public.waddle_paper_trades enable row level security;
-- no policies on purpose: only the service-role key can read or write.


-- Single-row snapshot of the bot's account state, so a redeployed / restarted
-- host (e.g. Railway, whose disk is wiped on every deploy) resumes exactly where
-- it left off instead of thinking it is flat.
create table if not exists public.waddle_bot_state (
  id          smallint primary key,        -- always 1
  state       jsonb not null,
  updated_at  timestamptz not null default now()
);

alter table public.waddle_bot_state enable row level security;
-- no policies: service-role key only.
