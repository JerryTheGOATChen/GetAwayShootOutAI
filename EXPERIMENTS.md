# Experiments: what we changed and what it did

A record of every batch of training runs, with its results, so we can see what works and what doesn't. Newest
entry at the bottom.

Each batch has a number and a nickname, and everything it makes goes in `runs/<number>-<nickname>/`: the best
agents, snapshots, logs (every run's reward and its breakdown), raw data, the chart, and the copy of your best run
it practised from. The batches are defined in `BATCHES` in `common/workers.py`; to try something new, add one at
the end there and an entry here, then `py -m common.run_batches <its name>`.

Race the best agents from different batches against each other:

    py -m sim.play --agent runs/01-from-the-start/ppo_tower.pt runs/02-your-run/ppo_tower_run.json runs/03-speed/ppo_tower.pt

## Fixed unless an entry says otherwise

- **Sim:** Tower, calibrated against recordings of the current game (lean, ground check, wall contact,
  checkpoint tracking on respawn), fixed lift schedule, lift crush on. Results from different sim versions are
  not comparable: when the physics changes, start the batches again.
- **Both methods:** the same network (53 inputs -> 64 -> 64 -> 3), the same reward (+0.1 per 112 px of route,
  -0.1 per second, +10 finish, -1 death, -5 and the episode ends after 20 s without progress), 50 million sim
  steps, seed 0. Both methods in a batch get the same settings, unless its entry says otherwise.
- **Saving:** `save_by` in `BATCHES`. `"reward"` keeps the agent with the highest reward; `"time"` keeps the
  fastest to the getaway (the highest reward until anything finishes). Learning and breeding always use the reward.
- **Practice from your run:** `practice` in `BATCHES`. Half the episodes the methods learn from start at a random
  moment of your best run instead of the start (genetic: each member also plays from one moment per generation,
  ranked on both rewards added up). Test runs always start at the start. Sim steps count the agents' own play
  only, not replaying your run up to the moment.
- **Snapshots:** the agent from every power-of-10 test run (PPO) or generation (genetic) is also kept, as
  `<method>_run<N>.pt` in the batch's folder, e.g. to show learning in a video.
- **Training time:** the PC is kept awake while training runs, and the batches run one after another, so the
  times are comparable.
- **Human reference:** your best run in the current sim (`runs/pb_Tower.json`, copied into each batch's folder as
  `your_run.json`): 42.42 s.

## What the numbers mean

- **Saved agent:** what's in `<method>_tower.pt`, the agent `sim.play --race` uses.
- **Fastest finish:** the quickest test run seen during training (the agent always playing its top-rated action).
- **Highest reward:** the best-scoring test run. Reward and time disagree when a run takes a faster route but
  dies on the way, or reaches further before stalling.
- **First finish / last improvement:** sim steps used when the agent first reached the getaway, and when its
  best time last improved.
- **Repeatability:** the genetic algorithm repeats exactly with the same seed. PPO does not, so PPO differences
  of under about 1 s between batches are noise.

## Entry template

    ## Batch N: Nickname (date)

    `runs/NN-nickname/`

    **Change:** what was changed and why.
    **Expected effect:** what we think it will do.

    | | PPO | Genetic |
    |---|---|---|
    | Saved agent | | |
    | Fastest finish | | |
    | Highest reward | | |
    | First finish | | |
    | Last improvement | | |
    | Training time | | |

    **What we learned:**

---

## Batch 1: From the Start (2026-10-03)

`runs/01-from-the-start/`

**Change:** none, the baseline. Every episode starts at the start. Saving by highest reward.

**Expected effect:** in earlier tests PPO stalled at the roof ledge (73%): the climb needs a short tap from the
top of a block up onto the ledge, and an agent that has learned to hold its lean never tries it. The genetic
algorithm got past it (36.16 s); since it repeats exactly, it should get exactly that again.

**Results:** as expected on both counts.

| | PPO | Genetic |
|---|---|---|
| Saved agent | 72.1% of the route, reward +1.30 (22.8M steps) | 37.16 s, reward +21.40 (50.0M steps) |
| Fastest finish | none | 36.16 s |
| Highest reward | +1.30 | +21.40 |
| First finish | none | 3.7M steps (79.28 s) |
| Last improvement | 72.9% of the route, flat from about 5M steps | 44.8M steps |
| Training time | 71 min | 53 min |

**What we learned:** PPO never got past the ledge in 50M steps (best 72.9%). The genetic algorithm reproduced the
earlier run to the step (first finish at 3,686,700 steps, 36.16 s at 44,801,826), which confirms it repeats exactly
and that nothing since changed it. Random changes to its weights stumble onto the tap; PPO's dice rolls don't.

---

## Batch 2: Your Run (2026-10-03)

`runs/02-your-run/` (its PPO agent's network and logs were overwritten on 2026-10-20; its exact run is kept as
`ppo_tower_run.json`, see "Test in the real game, part 2")

**Change:** half the practice episodes start at a random moment of your best run (42.42 s), for both methods.
Saving by highest reward.

**Expected effect:** PPO gets to practise the roof climb (and everything after it) while it still tries things
at random, so it finds the tap onto the ledge and finishes. In earlier tests with a slower run of yours (64.98 s)
this took PPO from stuck at 73% to 32.74 s, and the genetic algorithm finished in 35.10 s.

**Results:** practice from your run worked for both methods, PPO most of all.

| | PPO | Genetic |
|---|---|---|
| Saved agent | 32.84 s, reward +21.90 (41.3M steps) | 34.68 s, reward +21.72 (47.8M steps) |
| Fastest finish | 32.42 s | 34.68 s |
| Highest reward | +21.90 | +21.72 |
| First finish | 4.3M steps (50.82 s) | 12.4M steps (73.40 s) |
| Last improvement | 15.2M steps | 47.8M steps |
| Training time | 88 min | 67 min |

**What we learned:** PPO went from never finishing to finishing at 4.3M steps, and 99% of its last test runs
finished. Its best time stopped improving at 15.2M: with the normal reward, a faster finish is worth very little.
The genetic algorithm sat at the ledge (about 74%) until 12.4M steps, much longer than in batch 1 (each generation
is ranked on a different practice moment, which blurs the ranking), but then kept improving to the very end and
beat batch 1 (34.68 vs 36.16 s). It might improve further with more steps.

---

## Batch 3: Speed (2026-10-03)

`runs/03-speed/`

**Change:** the finish pays +1 more for every second left on the 90 s clock, so a second saved is worth 1.1
instead of 0.1. Runs that don't finish are scored as before. Saving by fastest time. PPO practises from your run
as in batch 2. The genetic algorithm practises from it only if that gave it a faster best time in batch 2 than
without it in batch 1 (decided automatically when batch 3 starts; its first line in the terminal says which).

**Expected effect:** with the normal reward, finishing 4 s faster is only worth +0.4 out of about 21, so the agents
learn to finish, not to hurry. With this, they should keep cutting time after they first finish: faster best
times than batch 2.

**Results:** PPO got faster but takes risks; the genetic algorithm got slower. Batch 3's genetic algorithm did
practise from your run (34.68 s with it in batch 2, 36.16 s without in batch 1).

| | PPO | Genetic |
|---|---|---|
| Saved agent | 31.42 s, 2 deaths (37.4M steps) | 38.34 s, no deaths (35.7M steps) |
| Fastest finish | 31.42 s | 38.34 s |
| Highest reward | +79.27 (31.74 s) | +72.93 (38.34 s) |
| First finish | 3.9M steps (54.80 s) | 2.7M steps (76.58 s) |
| Last improvement | 37.4M steps | 35.7M steps |
| Training time | 75 min | 51 min |

**What we learned:**
- PPO is the fastest of any batch, 31.42 s, about 1 s faster than batch 2, and it kept improving until 37.4M steps
  (batch 2's PPO stopped at 15.2M). But it plays riskily: its fastest run dies twice, its last test runs average
  1.5 deaths, and 91% of them finish (99% in batch 2). A death costs 1 and a respawn, while saving time now pays a
  lot, so risky shortcuts became worth it. About 1 s is at the edge of PPO's run-to-run noise, so this needs a
  second seed to be sure.
- The genetic algorithm reached the end of the route fastest of all (2.7M steps) but finished slowest of the three
  that practised: 38.34 s, against 34.68 s in batch 2. A guess, not yet checked: the finish bonus is so large
  (about +50 to +60) that every finisher scores far above every non-finisher, and the small differences in time
  between finishers count for little next to the noise from each generation's practice moment.
- Every agent except batch 1's PPO beats your 42.42 s in the sim. The fastest, batch 3's PPO, is 11 s quicker.

Charts: `runs/<batch>/compare.png` for each batch, `runs/compare_all.png` for all of them (PPO solid, genetic
dashed), and `runs/route_paths.png` for everyone's route on the map.

---

## Test in the real game (2026-10-09)

`real/replay.py`, `real/record.py`, recordings in `real/recordings/`

**Change:** no training. The saved agents' key presses from their sim runs, played into the real game (browser,
full screen) at the same times, starting when the script sees the drop-in landing.

**Results:**
- The start is spotted within about 2 frames of the game's clock (5 recordings, no false starts in 2 long clips of
  normal play).
- Every agent's run drifts away from its sim run within about 2 s, at the first landing, and never recovers.
- Replaying 30 s of your own recorded street jumps (tap chain included) in the sim stays within about 40 px of the
  real run the whole way. So the sim's physics holds up; the open-loop replay is what fails.

**What we learned:** the agents' play is far more fragile than yours. You stand still between moves, which wipes
out small timing errors; the agents make over 100 key changes a run, mid-air taps and chained jumps, each built
on the last. They were trained in a sim that repeats perfectly, so nothing taught them to recover from being one
frame off, and the game reads keys at 60 Hz while its physics runs at 50, so being a frame off is unavoidable.
A live agent that watches the screen would fix that, but locating the player in the level from the picture
(the background is fixed to the screen; inverting the camera's motion drifts about 300 px in 30 s) is unsolved,
so it is shelved.

---

## Batch 4: Jitter (2026-10-19)

`runs/04-jitter/`

**Change:** batch 2's settings, but in the episodes PPO learns from, every key change reaches the game late: 0-2
ticks for the whole episode, plus one more tick at a coin flip for each change (`Jitter` in `sim/env.py`). That is
the real game's timing: it reads keys once a frame at 60 Hz while its physics runs at 50 Hz, and the start can only
be spotted to about a frame. Test runs, and so the saved agent, stay exact. PPO only (the genetic algorithm ranks
members on single exact runs, so jitter would only add noise to it).

**Expected effect:** an agent that doesn't rely on tick-perfect timing, so its run, played blind into the real
game, stays on track longer than the 2 s of batches 2 and 3.

| | PPO |
|---|---|
| Saved agent | 34.02 s, reward +21.86 (43.5M steps) |
| Fastest finish | 34.02 s |
| Highest reward | +21.86 |
| First finish | 3.1M steps (40.16 s) |
| Last improvement | 43.5M steps |
| Training time | 88 min (the first 6 at half speed: Windows was throttling it as a background job) |

In the sim with late keys (`py -m real.robustness ... --seeing`, 100 runs each; the agent sees and reacts):

| | finishes | median time | deaths a run |
|---|---|---|---|
| 04 jitter PPO | 100% | 38.71 s | 0.1 |
| 02 PPO | 100% | 39.53 s | 0.7 |
| 03 speed PPO | 99% | 39.96 s | 1.9 |
| 02 genetic | 90% | 59.45 s | 2.0 |
| 03 genetic | 54% | 53.70 s | 1.5 |

Blind, its exact keys replayed with real-game timing errors (`py -m real.robustness`, 100 replays each; "on
track" = the jeans within 100 px on the game's screen, as measured in the real game):

| | stays on track (median) | in the level (100 px) |
|---|---|---|
| your PB | 3.4 s | 2.6 s |
| 02 PPO | 3.2 s | 2.1 s |
| 03 speed PPO | 2.2 s | 2.2 s |
| 02 genetic | 11.3 s | 8.3 s |
| 04 jitter PPO | 5.2 s | 2.5 s |

**What we learned:**
- Late keys in practice made a steadier agent: with late keys it dies 0.1 times a run instead of 0.7, and finishes
  a little faster than batch 2's agent does under the same delays, though its best exact time is 1.2 s slower.
  It also first finished sooner (3.1M steps against 4.3M).
- Blind, it is only a little better (5.2 s on screen, 2.5 s in the level). Any fast run is chaotic: in the sim,
  moving every key of your PB one tick later sends it 100 px off within 3 s. A run played blind can't survive the
  real game's timing however it was trained; only seeing and correcting can.
- The PPO agents already cope with late keys when they can see (99-100% finishes); the genetic agents much less
  (54-90%), the reverse of blind, where batch 2's genetic run is the most robust (11 s). Its moves are slower and
  simpler, so small timing errors grow more slowly, but its network never practised recovering.
- Its real-game trials didn't run: Windows locked the screen during the night (it can't be unlocked without you).
  It opens with exactly batch 2's keys and re-jumps on its first landing like it, the move that sent batch 2's
  run off in every real trial at 1.5 s (see below), so it will very likely go the same way.

---

## Test in the real game, part 2 (2026-10-20)

`real/trials.py`, `real/trials_report.py`, trials in `real/trials/`, probes in `real/probes/`, chart
`runs/real_game.png`

**Change:** no training. `real.trials` plays runs' keys into the real game by itself, many times, restarting from
the pause menu, and records Steve's jeans on the screen. `real.trials_report` compares each trial with its sim run
filmed with the game's camera, and the trials with each other. 44 trials of 4 runs, then 39 trials of probes.

**Results:**

| | trials | left its sim run at | trials stayed together |
|---|---|---|---|
| your PB (42.42 s) | 11 | 2.6 s (every trial 2.56-2.69) | 14.6 s (median pair) |
| 02 PPO (32.84 s) | 11 | 1.5 s (1.46-1.56) | 3.3 s |
| 03 speed PPO (31.42 s) | 11 | 1.5 s (1.47-1.57) | 7.2 s |
| 02 genetic (34.68 s) | 11 | 5.9 s (5.83-6.02) | 13.1 s |

Probes: batch 2's genetic run up to its landing at 5.70 s, then one jump 0.04-0.54 s after it. The velocity after
the jump that best fits each probe's real path, against the sim's:

| jump after the landing | sim | real game | real jump direction |
|---|---|---|---|
| 0.04 s | (1774, 653) px/s | (940, 1260) | 9 deg from upright |
| 0.10 s | (1735, 653) | (940, 1300) | 10 deg |
| 0.16 s | (1695, 653) | (980, 1240) | 14 deg |
| 0.28 s | (1616, 653) | (1180, 1080) | 29 deg |
| 0.54 s | (1444, 653) | (1400, 760) | 52 deg |

**What we learned:**
- **The real game is repeatable.** Played the same keys, it does the same thing for 13-15 s (your PB, the genetic
  run), to within 0 px for the genetic run's first 5 s. One trial in 11 of your PB split off at 3.6 s.
- **So the gap is in the sim, not in timing noise.** Every trial of a run left its sim run at the same moment,
  and shifting all the keys by -4 to +4 ticks doesn't help (`py -m real.physics_check ... --variants keys_shifted=N`).
- **The move the sim gets wrong: jumping again soon after a landing.** Land holding a lean key, and the lean starts
  again on the ground. The sim's jump then goes along the body, still tilted from the landing (56 deg); the real
  game's along a lean that builds up from upright again, about 9 deg + the lean from standing (chart:
  `runs/rejump_probe.png`). The jump's strength is the same (~1240-1320 px/s). Every run uses this move: your PB
  4 times, the agents 7-13 times, gaining 450-850 px/s of sideways speed each time (`py -m real.tilt_jumps`).
  The PPO agents start with it: land holding the other key and swap.
- **A second, smaller gap:** the full-lean jump from a standing start lands a little later and lower in the game,
  on a different side of the step near the start (probes `landA_*`, `landB_*`, which all left the sim at 1.5 s).
- Changing the sim so a lean started from a tilt jumps along a lean that ramps up from upright keeps it on the
  real runs 1-3 s longer (your PB 2.6 -> 5.5 s), and the 0.16-0.28 s probes to their ends, but no version fits
  every probe yet. The sim is unchanged: all the trained agents depend on it, so that is a decision for later.

**Lost:** while checking a code change after these tests, a 60 s test run of `ppo.train` without `--resume`
replaced batch 02's PPO files (its agent, log, raw episodes, checkpoint and first snapshot). Nothing else was
touched. Its exact 32.84 s run was rebuilt from the keys the real-game trials sent for it and is kept as
`runs/02-your-run/ppo_tower_run.json`; it replays to the same 32.84 s, and `sim.play --agent` and the `real.*`
tools take it in the agent's place. Its training curve survives only as pictures, kept as
`runs/02-your-run/compare_with_ppo.png` and `runs/compare_all_with_02_ppo.png` (redrawing the charts now leaves
it out). The trainers now refuse to start over an existing batch without `--resume` or `--fresh`.

---

## Rebuilding the sim from the real game (2026-10-20, morning)

`sim/world.py` (`Movement`, `ELEVATOR_HEAD_START`, `Game._update_elevators`), `real/replay.py` (timing),
`real/physics_check.py`, trials in `real/trials/2026-10-20 07-*`, chart `runs/sim_vs_real.png`

**Change:** no training. 59 more real-game trials (main runs and probes from a standing start), now also tracking
Steve's head, so his body angle can be read from every frame (`real/body.py`). Each candidate change to the sim was
tried against every trial with `real.physics_check`, which reports how long each trial stays on the sim's path.

**What was wrong, and the fixes:**

| | found from | fix | your PB stays on track |
|---|---|---|---|
| Start of the run | leans from the start began 0.12 s late in the game, later keys didn't | the game takes keys 0.16 s after the drop-in landing, not 0.04 | 2.6 s |
| Keys acting late | every run fitted better with all keys 2 ticks later | send keys 0.04 s early (`KEY_LEAD`) | |
| A lean starting in the air | switching keys at a jump: the sim's body swung 56 deg the other way in the air, the game's eased back upright | a jump leaves the ground at once (`jump_leaves_ground`) | |
| Centre of mass | landing tilted, the game's Steve rocks back before tipping forward; the sim's tipped forward at once | centre of mass at 0.3 of his height, not 0.5 (`centre_of_mass`) | 12.5 s (with the two above) |
| Lifts | the game's Steve stays on the floor when a lift stops; the sim's was thrown into a 0.4 s hop, and the lift stopped 0.3 s late | lifts move by position and carry their rider; 0.3 s later in their cycle (`lift_carries`, `ELEVATOR_HEAD_START`) | 14 s |

Centre of mass scan (with the jump fix): at 0.5 (the old) the main runs left the real ones after 1.4-6.6 s; at 0.3,
after 4.3-14.4 s; 0.2-0.33 did nearly as well, 0.15 and lower worse again. Changing the body's inertia by 30% either way made
everything much worse, so it stays the box's own. Things tried and left out because they fit worse: a shorter
ground check, waiting for real contact before a lean can start, a delay before a lean can start after landing,
friction 0.8 or 1.6, keeping only part of the speed at a jump or a landing.

**Results** (how long a real-game trial stays on its sim run; the last column is how long real trials of the same
run stay with each other, the most any sim could):

| | old sim | rebuilt sim | real vs real |
|---|---|---|---|
| your PB | 2.6 s | 14.0-14.4 s | 13.9 s |
| 02 genetic | 5.9 s | 13.2 s | 11.0 s |
| 02 PPO | 1.5 s | 7.8 s (2.3 s in one session) | 6.5-19 s |
| 03 speed PPO | 1.5 s | 5.9 s | 5.1 s |
| 04 jitter PPO | 1.6 s | 4.3 s | 3.5 s |

Checked on fresh trials it wasn't fitted to (`real/trials/2026-10-20 07-58-08`): your PB stayed on its sim run for
12.5 s in all three trials with the replay timing alone, and 14.0 s once the lifts were fixed.

**What we learned:**
- **Last night's diagnosis was half right.** The sim was wrong around re-jumps after landings, but not because the
  game aims those jumps differently: the sim let a lean start in the jump's own frame (so the body was locked at
  full tilt through the flight and landed at 56 deg, where the game's had eased back to about 10 deg), and its
  centre of mass made every tilted landing tip forward. With both fixed, the probes that showed "9 deg jumps"
  follow the sim.
- **Some of the gap was the replay, not the sim:** the start was 0.12 s early and every key 0.04 s late.
- **The lifts mattered for any run that rides one** (all of them): the hop when a lift stopped delayed the next
  lean by 0.4 s in the sim.
- **The limit now is the game itself.** It reads keys at 60 Hz with physics at 50 Hz, so even the same keys only
  give the same run for 13-15 s, and only 4-8 s for the PPO runs, which re-jump the instant they land. The sim
  follows each run about as long as the game repeats it.
- **Consequences:** most old agents no longer finish in the new sim (they used the sim's mistakes); they are kept in
  `runs/old-physics-2026-10-19/` and every batch is retrained. Your PB's keys replayed in the new sim only get
  halfway, so practice starts from the moments of your run instead (recorded by replaying it under the old physics,
  `OLD_MOVEMENT`, and restored exactly with `Game.restore`). Your real recordings (2026-10-09) can't stand in for
  it: replayed blind they die by 8% of the route.

---

## Batches 1-4 again, on the rebuilt sim (2026-10-20)

`runs/01-from-the-start/` to `runs/04-jitter/` (the old ones are in `runs/old-physics-2026-10-19/`)

**Change:** the same four batches, trained again on the rebuilt sim. Practice starts from the moments of your run
recorded under the old physics (its keys get only halfway in the new sim).

| best finish in the sim | PPO | Genetic |
|---|---|---|
| 01 from the start | none (72% of the route: stalls at the roof ledge, as before) | 38.98 s |
| 02 your run | 34.34 s | 36.82 s |
| 03 speed | 33.74 s | 42.84 s |
| 04 jitter | 36.06 s | (PPO only) |

Blind in the real game (`real.trials`, the same day):

| | stayed on its sim run | furthest on the real progress bar |
|---|---|---|
| 01 genetic | 7.1 s | 19% |
| 01 PPO | 6.1 s | 9% |
| 02 genetic | 11.6 s (the sim's timing-error replays predicted 11.6) | 22% |
| 02 PPO | 2.6 s | 33% |
| 03 PPO | 2.8 s | 8% |

**What we learned:**
- **Every batch is slower than on the old sim** (best 33.74 s against 31.42 s): the old sim's re-jump was a shortcut
  that the real game doesn't have.
- **Blind, none finishes the real game**, as expected: the game itself only repeats a run for about 14 s.

---

## Seeing the real game: a shadow sim (2026-10-20, afternoon)

`tools/render_level.py`, `real/locate.py`, `real/shadow.py`, `real/live.py`; tracks of the recordings in
`real/tracks/`; rehearsals in `runs/closed_loop_rehearsal.txt`

**Change:** the agent stops replaying blind and plays from a *shadow*: the sim, run alongside the real game and
kept on it by watching the screen.

- **Locating:** the level's art, rendered from the 2019 build sprite by sprite in the sim's coordinates
  (`tools/render_level.py`), is today's art and lines up with the colliders: in the first frame of a recording the
  camera came out at x = -15.663 m, the sim's start is -15.66. Each grab is matched against it around where the
  camera model expects the camera (masked to where there is art: not the parallax city, the HUD or Steve). An axis
  is used only when the match pins it down (the street gives nothing along it; lift shafts repeat), otherwise the
  camera model's guess. Lift cars and the helicopter are found separately; after a death it searches the respawn
  points. Fitted on the way: the camera eases at speed 5.13 across and 4.89 up and down (the 2019 value is 5) and
  rests with Steve's feet 2.156 m below the middle; 113.3 px/m at 1080p (the sim's viewer uses 112.4).
- **The shadow:** each grab rewinds it to the tick it shows, moves Steve 60% of the way to where he was seen, changes
  his speed by 3 m/s per metre of error, sets the lifts where they were seen, and replays the ticks since with the
  same keys. Errors under 2 cm are ignored, and up-down ones under 10 cm while standing (his measured feet sit
  2-4 cm above floors). A grab over 1.5 m off is ignored, unless four in a row agree: then it re-syncs.

**Results on your two recorded full runs** (keys read off the OBS key overlay, nothing tuned on the second):

| | 2026-10-02 23-41-14 (45.21 s) | 23-45-41 (two deaths at the end) |
|---|---|---|
| feet above the sim's floors, standing | 2-4 cm | |
| grabs with the camera from the art on both axes | 88% | 82% |
| shadow off before each correction: median | 5.5 cm | 6.5 cm |
| ... 95% of grabs | 22 cm | 28 cm |
| ... over 30 cm | 1.6% | 4.1% |

(The first recording's figures are from the tracker before its last fix, which masks the lift cars and
looks for Steve again after a death; the second's from the fixed one.)
| the same keys played blind | off at once (median 35 m) | off at once |

The whole loop (`real.live --watch`) dry-run with the first recording playing in real time as the screen: it
spotted the start itself, made 36 grabs a second, each applied about 5 ticks after the moment it shows, and the
shadow finished 0.27 s from the real time.

**Rehearsal in the sim** (`py -m real.shadow --sim-test`): a second sim stands in for the game, with keys landing
0-3 ticks late; the agent sees it only through grabs 25 times a second, 2 ticks old, with 2 cm and 2 deg of noise.
30 runs each, finished:

| | sees the truth | from the shadow |
|---|---|---|
| 04 jitter PPO | 30 (median 38.98 s) | 27 (38.94 s) |
| 02 PPO | 30 | 7 |
| 03 speed PPO | 28 | 3 |
| 02 genetic | 16 | 1 |
| 01 genetic | 5 | 1 |
| 03 genetic | 1 | 0 |

Harder grabs for batch 04 (8 runs earlier, 30 here): 25 a second, 4 ticks old, 5 cm / 5 deg: 26 of 30; 17 a
second: 22; 12 a second, 6 ticks old, 10 cm / 8 deg: 19. The shadow expecting the keys' average delay (2 ticks)
instead of none: 27 against 23.

**What we learned:**
- **Where Steve is can be read off the screen to a few cm**, everywhere on the route, by matching the game's own art.
  The earlier worry (the background is fixed to the screen) was half right: the city behind is a parallax layer,
  but the level itself is all there to match.
- **A corrected sim stays on the real run indefinitely**, where any blind replay leaves it within seconds.
- **The agent decides whether that is enough.** Only the agent trained with late keys plays as well from the
  shadow as from the truth; the others lose most runs to the shadow's few cm. Its failures: stuck in the top
  office or on the balcony after a fall.
- **Bug found:** since the rebuild the lifts move by position, so their bodies have no velocity and every agent
  of batches 01-04 saw `lift_vy` = 0 all the time. Fixed for new batches only (`lift_speed_seen`), so the trained
  agents keep seeing what they were trained with.

---

## Batch 5: Shadow (2026-10-20)

`runs/05-shadow/`

**Change:** batch 04 (practice from your run, late keys) with two more things for playing from a shadow: practice
episodes nudge the player every 2 ticks by about what the shadow's corrections do (3 cm, 0.15 m/s, 1.5 deg,
not up or down while standing; `Wobble` in `sim/env.py`), and the agent sees which way the lifts move
(`lift_speed_seen`). PPO only.
**Expected effect:** a little slower in the sim, and as steady from the shadow as from the truth.

**Results:** 50M steps in 66 min; first finish after 4.4 min (batch 04: 7.3), best 35.74 s (batch 04: 36.06).
Rehearsed like batch 04 (`runs/closed_loop_rehearsal.txt`):

| finished, of 100 | sees the truth | from the shadow |
|---|---|---|
| 05 shadow PPO | 100 (median 39.39 s) | 78 (39.31 s) |
| 04 jitter PPO | 100 (39.08 s) | 87 (39.24 s) |

(The same runs for both, seeds 2000 on. The first 30 of another set: 30 and 23 against 30 and 27.)

**What we learned:**
- ~~The nudges didn't help.~~ **Corrected 2026-10-22:** the shadow had a bug (a re-sync launched it at 5 m/s per metre
  it had been off; see "Closed loop in the real game"), and batch 05 set it off more often. With it fixed, both
  finish nearly every run from the shadow: batch 05 99 of 100, batch 04 97. So the nudges didn't hurt, and may
  help a little; 2 runs in 100 is too few to say.

---

## Batch 6: Shadow, genetic (2026-10-20)

`runs/06-shadow-genetic/`

**Change:** batch 05 for the genetic algorithm. Each member is ranked on 4 runs with late keys and nudges instead
of exact ones (2 from the start, 2 from moments of your run; in a generation every member gets the same noise),
so a member can't get by on perfect timing. It also plays once from the start exactly, for the log; the agent
saved is the generation's best-ranked member, when its exact run is the best so far. 120M steps: each generation
costs about 2.5 times one of batch 02's (`noisy_runs` in `common/workers.py`).
**Expected effect:** genetic agents that hold up when keys land late and the player is pushed around, as PPO's
did in batch 04.

**Results:** 161 generations in 117 min. The best member's average reward on the noisy runs stayed around 0 for
100 generations, then rose to +16 by generation 140 and stayed there (a finish is about +21). Its exact run from
the start finished in only 8 of the 161 generations. Best exact finish by any member: 38.28 s; the agent saved
(generation 151): 44.76 s.

| finished, of 100 | sees the truth | from the shadow |
|---|---|---|
| 06 shadow genetic | 57 (median 70.18 s, 238 deaths) | 10 (55.25 s); 46 with the shadow's re-sync fixed |
| the last generation's best-ranked member | 24 | 3 |
| for comparison, 02 genetic (of 30) | 16 (66.29 s) | 1 |

**What we learned:**
- **Ranking on noisy runs helps a little**: more finishes from the shadow than any genetic agent before, but still
  slow and still dying about twice a run. PPO stays far ahead.
- **For a member bred on noise, the exact sim is one more draw of the noise**, and not a lucky one: the best-ranked
  members finished most noisy runs while their exact run often stopped in the first 10% of the route.
- It was still learning slowly at the end (the noisy reward only took off after generation 100); more generations
  might help.

## Closed loop in the real game (2026-10-21 evening, fixes 2026-10-22)

`real/live.py`, `real/clip.py`; runs in `real/live/<date time>/`

**Before playing, two things the recordings didn't show:**
- **The screen's scale.** The live screen is 1920x1200, and Steve is about 6% bigger on it than in the 1920x1080
  recordings real.locate was fitted on: about 120 px/m, not 113.3 (the art matched best at 118-122 on saved
  trial frames; his jeans-to-head distance gave 120.4). `py -m real.live --calibrate` now measures it from full-size
  grabs of Steve standing at the start, with how far above his feet the camera rests (about 2.22 m here, 2.156 in
  the recordings), and every live run uses what it saved.
- **The helicopter.** Its animation, `Heli_MoveVertical` in the 2019 build, sets the helicopter body's local height
  to between -0.25 and +0.25 m (a 2 s loop), replacing the scene's 1.3 m. So in the game the body, with its floor,
  walls and the finish trigger, hangs 1.05-1.55 m lower than the sim's still copy. Both recordings agree: 1.09-1.58
  m lower, at the same x. In a dry run of the live loop the shadow followed your run to the door, then hit the
  sim's helicopter floor from below and fell. The sim is unchanged for now (every agent depends on it).
- **A stall in the grabbing.** Once, 0.4 s passed between grabs while Steve rode a lift; the tracker's guess of the
  camera assumed he had stood still, came out 3.7 m off, and it never found the camera again (the shadow got to 38%
  of the route). It now aims between "stood still" and "moved steadily" after a long gap, and searches wide enough
  for both: in the same dry run it lost 113 grabs instead of 1047.

**Calibrated** (`py -m real.live --calibrate`, Steve standing at the start for 3 s): 119.8 px/m (the art matched
best at 119.5-120 and worse either side), the camera resting 2.02 m above his feet.

**The failure, filmed** (`py -m real.clip blind`, then `pair`): batch 03's PPO, 33.74 s in the sim, played blind.
Within 4 s the real Steve is still at the first lift while the sim one rides it up; at 11 s he wanders the street.
Clip: `real/clips/2026-10-21 22-52-21 blind 03-speed ppo_tower pair.mp4`, GIF `runs/real_blind_03_ppo.gif`.

**Batch 04 PPO, closed loop, 3 runs** (`py -m real.live runs/04-jitter/ppo_tower.pt --runs 3 --clip`, about 18
grabs a second with filming):

| | how far in the real game | what stopped it |
|---|---|---|
| run 2 | **the roof, and the jump for the helicopter (about 90%), 40 s in** | landed on the real helicopter's roof, lower than the sim's, and the rotor killed him; tracking lost after |
| runs 1, 3 | the second lift (about 55%), 22 s in | 2 s late after slips at the first lift: walked into the empty shaft; nothing picked the run up again |

Blind, no agent had got past 33%. In run 2 the shadow stayed within about 10 cm of the real Steve for 40 s,
through both lifts (the second ran about 0.2 s behind the sim's schedule; the shadow took the lift where it was
seen). Clip of run 2 beside its shadow: `real/live/2026-10-21 22-53-24/run2 pair.mp4`, GIF
`runs/real_closed_loop_04.gif`.

**Fixed after (2026-10-22), no retraining:**
- **The helicopter hovers in the sim** (`HOVERS`, `hover_y` in `sim/world.py`, `Movement.helicopter_hovers`): its
  colliders and the finish trigger move with Heli_MoveVertical on the lifts' clock (the first recording's bob
  fitted 1.58 s into the clip at tick 0, the lifts' head start is 1.55 s). The sim's helicopter now ranges 36.42-
  36.92 m, the recordings' 36.39-36.88. The tracker expects it there and the shadow takes its seen height, like a
  lift's. `MOVEMENT_2026_10_20` keeps the old physics, in which every agent until now was trained (their times
  replay exactly); in the new sim the PPO agents still finish, about 1.2 s slower.
- **Deaths:** the game's death notice ("Player 1" and a skull, white, on the left for about 3 s) is matched in
  every grab (`real/kill_marker.png`, `death_marker`); when it appears the shadow dies at that tick too and
  respawns where and when the game does. The tracker searches the respawn points when lost, and with two Steves on
  screen (his body stays where he died) takes the one nearest the shadow's. A shadow that "finished" while the
  real Steve died is put back. Replaying run 1's film through the loop: the notice came at 21.2 s, both
  respawned in the office at 23.0 s, and the shadow followed him within about 10 cm after.
- **Re-sync:** a shadow lost by over 1.5 m jumps to where four grabs in a row agree he is, but it also got a speed
  of 5 m/s for each metre it had been off: 3.4 m behind your jump to the roof (a dry run), it was launched off the
  tower at 17 m/s and never caught again. It now takes the speed the grabs show, compares where they saw him (not
  the errors), and catches up its checkpoints (the agent steers by the next three).
- **Restoring a moment** didn't restore whether the player's body was in the world: a shadow rewound across a
  death could freeze. Fixed in `Game.restore`.
- **The loop** kept going only until the shadow finished; it now plays on 3 s after, in case the real run hasn't.
- **Speed:** finding Steve takes 4 ms instead of 8.6 (one half-size copy of the grab, the head looked for near
  the jeans; same answers on 272 frames of your recording).

Dry runs with your 2026-10-02 run as the screen now finish in the shadow at 45.44-45.80 s (real 45.21 s),
33-44 grabs a second. Rehearsal again (100 runs, seeds 2000 on; deaths now told to the shadow 6 ticks late, as
the notice is):

| finished, of 100 | sees the truth | from the shadow, old helicopter | from the shadow, hovering |
|---|---|---|---|
| 04 jitter PPO | 100 | 97 | 93 |
| 05 shadow PPO | 100 | 99 | 95 |
| 06 shadow genetic | 57 | 46 | 45 |

With the helicopter where it really is, the agents die more at the end (33 deaths in 100 runs for batch 04,
against 11): they were trained to jump into one 1.3 m higher. That is for the retraining.

**What we learned:**
- **A closed-loop agent gets much further in the real game than any blind one:** 90% against 33%.
- **The sim's last metre was wrong all along.** The getaway helicopter hangs 1.05-1.55 m lower than the sim had
  it; no blind run ever got there to show it.
- **Most of the shadow's failures in rehearsal were its own bug**, not the agents': 87 and 78 of 100 became 97
  and 99.
- **Still open:** an agent that arrives late for a lift walks into the empty shaft (it never learned to wait for
  the next one); jumps just after landing still go off about 0.3 s differently in the game (the shadow corrects
  it).

---

## Conclusions

1. **Exploration was the real problem, not the algorithm.** From the start, PPO stalls at the roof ledge (72.9%)
   because it stops trying the short tap the climb needs; the genetic algorithm's random weight changes find it.
2. **Practising from a human run fixes it.** Starting half the episodes at moments of your 42.42 s run took PPO
   from never finishing to finishing at 4.3M steps, and to the best normal-reward time (32.42 s).
3. **The reward decides what "good" means.** Paying for seconds saved made PPO faster (31.42 s) but reckless
   (2 deaths in its best run, 91% of runs finish instead of 99%), and made the genetic algorithm slower.
4. **Every agent but batch 1's PPO beats your 42.42 s in the sim**, the best by 11 s.
5. **Agents find the sim's mistakes.** The real game turned out repeatable, and every run left its sim run at the
   same move: jumping again right after a landing, which the sim lets go much further sideways than the game does.
   Every agent uses it, 7-13 times a run. A slightly wrong sim is a target for an optimiser.
6. **Training for timing errors helps an agent that can see, not one playing blind.** Late keys in practice made a
   steadier agent (0.1 deaths a run with late keys, against 0.7), but no fast run survives blind: a single tick
   of difference grows into a different run within seconds. In the real game, an agent has to see.
7. **An agent can see the real game.** Matching the level's own art (rendered from the 2019 build) finds Steve to a
   few cm, and a sim kept on the real run by it lets an agent play closed loop: batch 04 got to the last jump (about
   90% of the route), where blind runs stopped at 33% at most.
8. **The real game shows the sim's mistakes only where an agent gets to.** The getaway helicopter was 1.05-1.55 m too
   high in the sim all along (its animation was left out); the first closed-loop run to reach it died on its rotor.

## Next, if picked up again

- **Retrain for the hovering helicopter:** `py -m common.run_batches 07-helicopter` (batch 05's settings in today's
  sim, about 70 min), then `py -m real.shadow --sim-test runs/07-helicopter/ppo_tower.pt --runs 30`, and in the
  real game `py -m real.live --calibrate` and `py -m real.live runs/07-helicopter/ppo_tower.pt --runs 5 --clip`
  (leave the game in a Tower run, full screen with the CrazyGames bar hidden).
- **Waiting for a lift:** an agent that misses a lift walks into the empty shaft. Practice never puts it there late
  (it always starts from your run's moments, with the lifts on their schedule); how to teach it without shifting
  the lifts is open.

- **Seeds:** every batch ran once. Run each 3-5 times with different seeds and compare averages, especially
  batch 3's 1 s gain for PPO, which is within PPO's run-to-run noise.
- **Exploration without your run:** try a curiosity bonus (e.g. RND) or Go-Explore in batch 1's setting, to see
  whether PPO can find the tap on its own.
- **Fix the re-jump in the sim:** pin down the rule with more probes from a standing start (`real/probes/`,
  `py -m real.probe_fit`), then retrain. With the sim's biggest gap gone, blind replays should stay with the
  real game about as long as real trials stay with each other (13-15 s).
- **Other maps:** Street, Train and Town are already in `sim/maps/`.
