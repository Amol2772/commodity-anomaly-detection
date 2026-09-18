# Model Card: Commodity Anomaly & Volatility Detection Pipeline

I wrote this the way a bank's model risk team would actually expect a model like this to be
documented — not a generic AI-ethics checklist, but the real practice used at financial
institutions (the Fed's SR 11-7 guidance, or the UK's PRA SS1/23) for writing down what a
model is for, what it isn't for, and where a person needs to stay in the loop. It felt like
the more honest and more relevant framing for something built with a trading desk in mind.

## What this is actually for

It's a decision-support tool. It flags days where a commodity's price or volatility behaved
unusually, so a human can go look at them — that's it. It's not meant to:

- Trigger a trade or a position change on its own
- Run without someone reviewing what it flags
- Be the only input behind a risk decision
- Be trusted on anything other than the three markets it was built and tested on (gold,
  silver, WTI crude)

The backtest notebook exists because "could you actually trade this?" is the obvious next
question an interviewer or a risk manager would ask, and I'd rather answer it honestly than
avoid it. It's not meant to double as evidence that this should be used as a trading strategy.

## How it works, briefly

Three separate detection methods (a robust rolling z-score, Isolation Forest, and `ruptures`
changepoint detection), cross-checked against an out-of-sample SARIMAX forecast. Full detail
is in the README.

## About the data

Daily prices for gold, silver, and WTI crude from Yahoo Finance, 2015 onward. Two things worth
knowing: silver has noticeably more zero-volume days than the other two (around 6% versus
roughly 1% and 0.2%), and continuous futures data like this can sometimes carry artifacts from
how contracts roll over — I checked the one dramatic case that came up here (WTI's 2020
negative price) and confirmed it was a real historical event, not a data glitch.

## Where I'd push back on my own results

- **The thresholds aren't rigorously optimal, and I don't want to imply they are.** I tuned
  `Z_THRESHOLD` in `project_config.py` by sweeping a handful of values and checking them
  against known events — a sensible way to pick a starting point, not the same as a proper
  validation study.
- **Isolation Forest's contamination rate doesn't perfectly match reality.** I set it to flag
  2% of days, but the z-score baseline naturally flags closer to 1.3–1.6%. I didn't hide that
  mismatch — it's shown directly in the dashboard's comparison table.
- **This catches sudden moves, not slow-building risk.** If a market quietly deteriorates over
  several months rather than moving sharply in a day, none of these three methods would
  necessarily flag it, even if a human analyst would consider it important.
- **It's only as good as the history it learned from.** Every threshold here reflects
  2015–2026 market behaviour. A genuinely new kind of shock — new market participants, new
  structure, something that's never happened before — is exactly the kind of thing a model
  built on history isn't well-positioned to catch. That's true of basically any statistical
  model, but I'd rather say it outright than leave it implied.
- **The backtest sample is small.** A few dozen trades per commodity over 11 years isn't
  enough to be confident either way about whether the underlying signal is real.

## Who should be in the loop

This should never act on its own. The way I'd want it used: a specific person reviews each
newly flagged day against what was actually happening in the market at the time — which is
exactly what I did by hand for the two big events this project catches — before anyone acts
on it.

## What happens as markets change

The settings I landed on (a 60-day window, a 4.0 threshold, Isolation Forest flagging roughly
2% of days) were tuned against 2015–2026 data. That's a reasonable place to start, not a
number I'd assume stays correct forever — markets shift, and what counts as "unusual" today
might not in a few years. I'd want these revisited periodically, and I'll say plainly: this
version doesn't have any automated way of noticing when they've gone stale. Building that
would be the natural next step for turning this into something production-grade, and it's
outside what I set out to do here.
