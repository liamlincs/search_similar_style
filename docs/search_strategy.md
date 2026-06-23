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

This prevents local matches such as sleeves or scarves from being hidden by
unrelated global/accessory heuristics.

## Tuning Guidance

- Increase `region_crop_code_prior_boost` if a correct region candidate appears
  in the `region=` log but is still not returned.
- Lower `region_crop_code_prior_min_score` only if correct region candidates are
  consistently below the threshold; lowering it too much will increase false
  positives.
- Keep `exact_region_rescue_enabled=false` unless exact same-color matching also
  needs the region rescue behavior.
- Do not add one-off style-code rules unless the failure is caused by bad data
  or a known catalog labeling issue.
