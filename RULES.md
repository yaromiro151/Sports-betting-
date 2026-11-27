# Prop rules bank

The working rules for NBA prop cards. Every rule here was earned by a graded result, and the
season doc (NBA Props Card) holds the full write-ups. Bet logs live in `history/dk_bets.csv`.
Update this file whenever a night's grading confirms, breaks or adds a rule.

_Last updated: Nov 25, 2026 (after grading Nov 24: 19 NBA bets, $749 staked, $2,994.70 paid, +$2,245.70)._

## Core rules (each confirmed again on Nov 24)

1. **Center/star rebounds are the backbone**, set one rung below the average. Nov 24: 5 of 7
   (Franz 6+ and 8+, Carter 9+, Ausar 8+, Dybantsa 7+ hit). Both misses broke a rule: Reed 8+ was
   above his 6.8 average, and Jokic 12+ was on a favorite that lost by 19.
2. **No wing or guard rebounds.** 0 of 3 (Suggs, Black, Gordon 5+).
3. **Threes only from high-volume shooters** (7+ attempts a game) at rungs they clear most nights.
   Low-volume threes: 0 of 3 (Hauser twice, Jenkins). Even high-volume shooters miss on stretched
   rungs (Bane 5+, Murray 4+).
4. **Assists only from the primary handler.** (Exception that worked: a role player's trend-ticket
   assists at his floor, like Queta 2+ and Braun 3+, both hit.) Secondary handlers and bigs went 1 of 4 (Pritchard,
   Kyshawn George, Black missed). Gafford 4+ and 5+ assists missed too.
5. **No steals or blocks, ever.** 0 of 4 on Nov 24 (Murray 2+ steals twice, George 2+ and 3+,
   Jokic 1+ block).
6. **Back a healthy star's points against a bad defense.** Tatum 25+/26+, Davis 20+ and George
   18+ hit. The misses were all Denver stars in an upset (rule 9).
7. No leg on more than 2 tickets; 4–6 legs a ticket.
8. Questionable stars are swaps, not legs. Check the injury report again within 30 minutes of
   tip (Cade was scratched at 6:36pm for a 7pm game).

## New / adapted rules (from Nov 24)

9. **Underdog-upset check (new).** When the game model makes the underdog 4+ points closer than
   the DK spread, don't stack the favorite's star overs. Nov 24: the model had DEN -3.7 against
   DK's -8.5 and flagged Dallas ML at +251. Dallas won 120-101, and Denver star overs went 0 of 8.
10. **Live role-player ticket (adapted).** Role-player points stay off pregame cards. They're
    allowed **live**, once the game script is clear, at low rungs (at or under the player's recent
    floor), for the team in control. Nov 24, Dallas role players live: 13 of 17, including both
    winning tickets ($1,350 and $960). The rungs above their recent form (Martin 15+ and 18+)
    missed, while 12+ and lower hit.
11. **Unders are singles, not parlay legs (adapted).** An overpriced over is a reason to skip the
    over, not to put the under in a parlay. Unders went 4 of 10 as legs (Trae, Queta, Pritchard,
    Carter and Flagg unders lost).
12. **Tough-matchup assists drop a rung.** Franz 7+ assists missed against Detroit, the #3
    assist defense, but Franz 6+ assists (one rung lower) hit on the same night. Against a top-5
   assist defense, use one rung lower or skip it.

## Model notes
- Prop model: `sportsedge/props.py` (negative binomial on verified ESPN box scores). The rules
  above are applied in `apply_rules` where they can be checked automatically.
- Game model: `python -m sportsedge card` gives the spread view that rule 9 uses.

## NFL rules (started Nov 25, 2026; ungraded until Week 12 settles)

1. **Volume only:** starting QBs (25+ attempts), RBs with 12+ carries, receivers with 6+ targets.
   No backups or gadget players, even with big model edges (those are thin samples).
2. **Wind:** outdoors at 20+ mph, no passing, receiving-yards or reception overs. Rushing overs
   and passing unders are fine. Check the forecast on game day; domes and closed roofs don't count.
3. **Upset check (NFL version):** when the game model makes the underdog 4+ points closer than the
   spread, skip the **favorite's rushing** overs (a trailing favorite throws more and runs less).
4. **A main-line edge of 15% or more means missing news** (role change, injury, weather), not a bet.
5. **Anytime TDs and unders are singles**, never parlay legs. Unders only when the hit rates agree.
6. **Questionable QBs put their whole passing game on the swap list** (e.g. Darnold, Week 12).
7. Re-check inactives 90 minutes before kickoff (NFL inactives come out then).

### NFL lesson, Nov 26 (PHI 39, DAL 38 after trailing 21-3)
8. **A big lead by a bad defense is not safe.** Dallas (29th in points allowed) led 21-3 and lost.
   Live player props still went 11-4 (Javonte carries/yards, Lamb, Saquon unders), but the projected
   final was badly wrong. When the leading team has a bottom-5 defense, add the **trailing team's
   WR1 over** live (DeVonta Smith went 11-200-2) instead of assuming the trailing offense is dead,
   and don't fade the leading QB's yards: the shootout keeps him throwing.
9. **Live NFL: carries and attempts are steadier than yards.** Bet volume first (Javonte 21+/22+
   carries hit easily), yards second.
