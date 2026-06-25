# Image Search Strategy

This project uses a two-scope search policy rather than treating every image as
one flat similarity problem.

## Match Modes

- `similar_style`: default mode. Prefer local structure, pattern, shape, stripe
  layout, and texture. Color is a weak signal because model photos and catalog
  photos often use different colorways.
- `exact`: conservative mode. Keep color and full-image consistency stronger,
  and avoid rescuing loosely similar region candidates into the final result.

The API accepts an optional `match_mode` form field. If it is omitted, the value
from `search.match_mode` is used.

## Search Scopes

- `full_context`: used when the user searches the whole uploaded image. It is
  best for overall category, scene text, logo, and broad visual style.
- `region_primary`: used when the user provides a crop box. The crop is treated
  as the primary search target. Full-image-style heuristics must not override a
  strong region match.

For model photos, `region_primary` is usually more reliable than whole-image
search when the target is a sleeve, collar, hat, scarf, or other local detail.

## Ranking Policy

For region searches:

1. Run the normal recall pipeline.
2. Run region-specific recall with lower color dependency.
3. Convert high-scoring region candidates into code-level priors.
4. Suppress accessory/hat-style candidate injection when region recall is
   already confident.
5. Rescue high-scoring region codes into the final result list if generic
   post-processing pushes them out.

In `similar_style:region_primary`, accessory/hat injection is also suppressed
when the crop already has accent-pattern candidates. This keeps sleeve, collar,
and scarf crops from being reinterpreted as hats just because the crop is close
to square.

This prevents local matches such as sleeves or scarves from being hidden by
unrelated global/accessory heuristics.

Small region boxes are automatically expanded before feature extraction. The
user-drawn rectangle is treated as an anchor, not as the final embedding image.
This keeps CLIP/hybrid semantics from being dominated by nearby labels, printed
text, or a single tiny patch with too little garment context.

If the original user crop is very small and had to be expanded aggressively,
the search enters a stricter region mode. In that mode, sleeve/accent/accessory
pattern injections and scene-text rescue are suppressed so the final ranking
stays closer to direct region recall instead of being rewritten by generic
local-pattern heuristics.
That strict-small path also uses query weights with lower stripe emphasis and
higher shape emphasis, so collar-like local crops do not collapse into generic
striped trim candidates as easily.

Very wide strip-like crops such as collars, plackets, waistbands, or hems are
handled separately. Their expansion height is capped so auto-expansion does not
pull in unrelated chest graphics, and accent-pattern injection is disabled when
the query is already in strip mode.

For strip-like region queries, region recall can also run with query multicrop.
This lets left/right/top local subviews of a collar or placket participate in
matching, which is useful when the user crop still contains some body fabric or
decorations that are not present in the flat-lay standard image.

## Tuning Guidance

- Increase `region_crop_code_prior_boost` if a correct region candidate appears
  in the `region=` log but is still not returned.
- Lower `region_crop_code_prior_min_score` only if correct region candidates are
  consistently below the threshold; lowering it too much will increase false
  positives.
- Keep `exact_region_rescue_enabled=false` unless exact same-color matching also
  needs the region rescue behavior.
- Tune `region_crop_context_pad_ratio` and `region_crop_context_min_area` when
  small crops are still text-biased; raise them for more garment context, lower
  them if region search becomes too broad.
- Tune `region_crop_strict_small_max_orig_area` and
  `region_crop_strict_small_min_expand_ratio` when very small label-like crops
  should stay closer to raw region recall and avoid sleeve/accent overrides.
- Tune `region_crop_wide_strip_max_h` when wide collar-like crops still include
  too much body area after expansion, and `region_crop_disable_accent_when_strip`
  if strip queries should ignore incidental logos or chest patches.
- Tune `region_strip_query_crop_ratio` when strip queries still miss target
  collar/placket matches; lower it to focus more locally, raise it to keep more
  context.
- Do not add one-off style-code rules unless the failure is caused by bad data
  or a known catalog labeling issue.
