# Prop rules bank

The working rules for NBA prop cards. Every rule here was earned by a graded result, and the
season doc (NBA Props Card) holds the full write-ups. Bet logs live in `history/dk_bets.csv`.
Update this file whenever a night's grading confirms, breaks or adds a rule.

_Last updated: Nov 25, 2026 (after grading Nov 24: 18 bets, $689 staked, $2,310 paid, +$1,621)._

## Core rules (each confirmed again on Nov 24)

1. **Center/star rebounds are the backbone**, set one rung below the average. Nov 24: 5 of 7
   (Franz 6+ and 8+, Carter 9+, Ausar 8+, Dybantsa 7+ hit). Both misses broke a rule: Reed 8+ was
   above his 6.8 average, and Jokic 12+ was on a favorite that lost by 19.
2. **No wing or guard rebounds.** 0 of 3 (Suggs, Black, Gordon 5+).
3. **Threes only from high-volume shooters** (7+ attempts a game) at rungs they clear most nights.
   Low-volume threes: 0 of 3 (Hauser twice, Jenkins). Even high-volume shooters miss on stretched
   rungs (Bane 5+, Murray 4+).
4. **Assists only from the primary handler.** Secondary handlers and bigs went 1 of 4 (Pritchard,
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
    assist defense. Against a top-5 assist defense, use one rung lower or skip it.

## Model notes
- Prop model: `sportsedge/props.py` (negative binomial on verified ESPN box scores). The rules
  above are applied in `apply_rules` where they can be checked automatically.
- Game model: `python -m sportsedge card` gives the spread view that rule 9 uses.
