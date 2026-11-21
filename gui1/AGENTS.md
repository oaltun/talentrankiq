there are multiple repos here:

gui_old: ./talentrankiq_gui_tauri2/gui_tauri2 (./talentrankiq_gui_tauri2 has ai agent support for gui_tauri2)
gui: ./talentrankiq_gui
cli: ./talentrankiq_cli
oculix: for smoke testing
git: for git actions that work on both gui and cli

read ./**/*.md (except prompt.md files, they are not general).

i wanna see how effective ai agents can be in developing this gui using tauri test and accessibility stuff. Use them to check whether what you want is created actually, before claiming me that you have done your work:
 - playwright e2e testing. installed, configured, a simple test exists. write e2e tests before you do your thing. `npm test` runs those tests. 
 - playwright-cli. installed. check `playwright-cli --help`.

 ## how gui will use cli (triq)?

`cli` creates a binary called triq (talentrankiq/target/release/triq). that is the cli gui will spawn as subprocess and control/monitor through stdin/stdout jsonl. triq can tell you how to use its stdin/stdout jsonl.

JSON Schema (for validation, codegen, docs):
triq protocol-schema > protocol-schema.json

TypeScript types (for TS consumption if needed):
triq protocol-ts > protocol.ts

Or write directly to files:
triq protocol-schema --output ./gui/src/protocol-schema.json
triq protocol-ts --output ./gui/src/protocol.ts

there is also a protocol_version field (semver), gui can check whether it knows the protocol_version, and reject if the major version is different. patch and minor should be ok.

export these and checkout.

## no test mocks
e2e and oculix... tests should not bypass cli, etc. they should do everything the actual app does to the tee. DONT MOCK ANYTHING. When you MOCK, its like cheating. it makes these tests not-valuable. i want gui tests to test the cli, and gui-cli integration too.

## development pseudocode:

if adding new feature:
    write failing tests that would test the feature you want
Do your code changes
while True:
    Run the cli tests: 
        - cd to the cli dir
        - run `bash test_fast.sh`
        - Fix any test failures
    Run the gui tests:
        - cd to the gui dir
        - then run `bash test_fast.sh`
        - Fix any test failures (If failure is a tauri-pilot scenario, checkout talentrankiq_gui/tauri-pilot-failures. There will be a screenshot (image) of the failure screen)
    Break if there were no test failures



## prune as much code as you can
as we add / remove / change features, the code becomes a mess. if you are replacing / changing something , remove the old version of the code. keep the existing code amount minimum.


## test jobad and cvs
talentrankiq_cli/test-cv/Job2000CVs/ has actual CVs in cv/ and a jobad description in jobad.txt . you can select those and try the progress bars, etc. quick and deep embedding vectors may be cached, which will change the progress bar behavior. you may want to turn caching on / off and try in both cases.

## Database Concurrency (CLI)

**Current issue**: `DbConnection(Mutex<Connection>)` serializes all SQLite access. In WAL mode, SQLite supports multiple concurrent readers + one writer, but the mutex prevents this.

**Watch path contention**: `ScanWorker::scan` holds the DB lock while:
1. Scanning filesystem (2000 files)
2. Inserting candidate rows one by one
This blocks `get_job_detail` (and other readers) for 10–30s, preventing workspace from loading.

**Fix strategy**:
1. **Read pool**: Use a separate read-only connection pool (or `Mode::ReadOnly` connections) for SELECT queries (`get_job_detail`, `list_jobs`, etc.)
2. **Writer connection**: Keep a single writer for INSERT/UPDATE/DELETE
3. **Per-row locking in scans**: Collect file list *without* DB lock, then for each file: acquire writer → insert → release. Use `busy_timeout` (e.g. 5s) to avoid writer starvation.
4. **WAL + busy_timeout**: Enable WAL mode (`PRAGMA journal_mode=WAL`) and set `PRAGMA busy_timeout=5000` so readers don't block writers and vice versa.

**Files to change**:
- `talentrankiq_cli/lib/libtriqui-cli/src/serve/mod.rs` — `DbConnection` type, `open_connection_at`
- `talentrankiq_cli/lib/libcv-pipeline/src/lib.rs` — `ScanWorker::scan` loop granularity

## Comparator artifacts

Comparator artifacts are gated/licensed/monetized. For now, the GUI loads them from local disk only (file picker). Remote download will be added later.


# design skills

use the following design skill:
```
apple1@Apple1s-MacBook-Pro talentrankiq_gui % npx ux-ui-agent-skills@3.1.0 init          # full kit into the current folder

Need to install the following packages:
ux-ui-agent-skills@3.1.0
Ok to proceed? (y) 

Installing full kit → /Users/apple1/Documents/repo/act/ik/activeik/talentrankiq/talentrankiq_gui/#

  + #/CLAUDE.md
  + #/tokens/blur.json
  + #/tokens/borders.json
  + #/tokens/breakpoints.json
  + #/tokens/colors.json
  + #/tokens/data-viz.json
  + #/tokens/gradients.json
  + #/tokens/motion.json
  + #/tokens/opacity.json
  + #/tokens/shadows.json
  + #/tokens/sizing.json
  + #/tokens/spacing.json
  + #/tokens/states.json
  + #/tokens/theming.json
  + #/tokens/typography.json
  + #/components/atoms.md
  + #/components/data-display.md
  + #/components/data-viz.md
  + #/components/feedback.md
  + #/components/forms-advanced.md
  + #/components/icon-system.md
  + #/components/molecules.md
  + #/components/navigation.md
  + #/components/organisms.md
  + #/components/overlays.md
  + #/components/templates.md
  + #/taste/aesthetic-systems.md
  + #/taste/design-taste.md
  + #/taste/motion-choreography.md
  + #/design-systems/catalog.md
  + #/design-systems/crosswalk.md
  + #/design-systems/interop-protocol.md
  + #/design-systems/library.json
  + #/frameworks/adapter-protocol.md
  + #/frameworks/adapters/angular.md
  + #/frameworks/adapters/astro.md
  + #/frameworks/adapters/bootstrap.md
  + #/frameworks/adapters/chakra.md
  + #/frameworks/adapters/css-in-js.md
  + #/frameworks/adapters/flutter.md
  + #/frameworks/adapters/jetpack-compose.md
  + #/frameworks/adapters/mantine.md
  + #/frameworks/adapters/mui.md
  + #/frameworks/adapters/qwik.md
  + #/frameworks/adapters/react-native.md
  + #/frameworks/adapters/solid.md
  + #/frameworks/adapters/svelte.md
  + #/frameworks/adapters/vanilla-css.md
  + #/frameworks/adapters/vue.md
  + #/frameworks/adapters/web-components-lit.md
  + #/frameworks/nextjs.md
  + #/frameworks/react-tailwind.md
  + #/frameworks/swiftui.md
  + #/accessibility/aria-patterns.md
  + #/accessibility/cognitive.md
  + #/accessibility/i18n-rtl.md
  + #/accessibility/vision.md
  + #/accessibility/wcag-aaa.md
  + #/accessibility/wcag-checklist.md
  + #/workflows/design-qa.md
  + #/workflows/design-review.md
  + #/workflows/design-to-code.md
  + #/workflows/figma-integration.md
  + #/workflows/governance.md
  + #/workflows/performance.md
  + #/workflows/prototyping.md
  + #/workflows/redesign-audit.md
  + #/workflows/token-build.md
  + #/content/voice-tone.md
  + #/scripts/_browser.mjs
  + #/scripts/_color.mjs
  + #/scripts/_targets.mjs
  + #/scripts/accuracy_report.mjs
  + #/scripts/axe_audit.mjs
  + #/scripts/build_tokens.mjs
  + #/scripts/check_links.py
  + #/scripts/check_no_emoji.py
  + #/scripts/contrast.py
  + #/scripts/design_systems.py
  + #/scripts/gate.mjs
  + #/scripts/library.mjs
  + #/scripts/lint_hardcodes.py
  + #/scripts/lint_intent.mjs
  + #/scripts/lint_native_select.py
  + #/scripts/lint_taste.py
  + #/scripts/measure_install_paths.py
  + #/scripts/measure_render.mjs
  + #/scripts/observe.mjs
  + #/scripts/pin_version.mjs
  + #/scripts/render_framework_source.mjs
  + #/scripts/scaffold_component.py
  + #/scripts/screenshot_docs.mjs
  + #/scripts/slop_tells.mjs
  + #/scripts/stats.mjs
  + #/scripts/taste_audit.mjs
  + #/scripts/update_homebrew_formula.mjs
  + #/scripts/validate_agents_surface.py
  + #/scripts/validate_component_spec.py
  + #/scripts/validate_contrast.py
  + #/scripts/validate_instruction_surface.py
  + #/scripts/validate_template.py
  + #/scripts/validate_theme_refs.py
  + #/scripts/validate_tokens.py
  + #/scripts/verify_focustrap.mjs
  + #/scripts/verify_interactive.mjs
  + #/scripts/verify_keyboard.mjs
  + #/scripts/verify_mobile_first.mjs
  + #/scripts/verify_overflow.mjs
  + #/scripts/verify_reduced_motion.mjs
  + #/scripts/verify_responsive.mjs
  + #/scripts/verify_rtl.mjs
  + #/scripts/verify_screen_economy.mjs
  + #/scripts/verify_spacing.mjs
  + #/scripts/verify_states.mjs
  + #/scripts/verify_target_size.mjs
  + #/scripts/verify_thai.mjs
  + #/.claude/skills/a11y-audit/SKILL.md
  + #/.claude/skills/apply-aesthetic/SKILL.md
  + #/.claude/skills/brandkit/SKILL.md
  + #/.claude/skills/critique/SKILL.md
  + #/.claude/skills/data-dashboard/SKILL.md
  + #/.claude/skills/design-code/SKILL.md
  + #/.claude/skills/design-component/SKILL.md
  + #/.claude/skills/design-doctrine/SKILL.md
  + #/.claude/skills/design-qa/SKILL.md
  + #/.claude/skills/design-review/SKILL.md
  + #/.claude/skills/design-screen/SKILL.md
  + #/.claude/skills/design-tokens/SKILL.md
  + #/.claude/skills/figma-integration/SKILL.md
  + #/.claude/skills/gate/SKILL.md
  + #/.claude/skills/governance/SKILL.md
  + #/.claude/skills/grill-me/SKILL.md
  + #/.claude/skills/image-to-code/SKILL.md
  + #/.claude/skills/migrate-design-system/SKILL.md
  + #/.claude/skills/observe/SKILL.md
  + #/.claude/skills/performance/SKILL.md
  + #/.claude/skills/prototype/SKILL.md
  + #/.claude/skills/redesign/SKILL.md
  + #/.claude/skills/scaffold-project/SKILL.md
  + #/.claude/skills/ship/SKILL.md
  + #/.claude/skills/token-build/SKILL.md
  + #/.claude/skills/ux-writing/SKILL.md
  + #/.claude/rules/accessibility.md
  + #/.claude/rules/brand-and-operations.md
  + #/.claude/rules/components.md
  + #/.claude/rules/frameworks.md
  + #/.claude/rules/review-and-research.md
  + #/.claude/rules/tokens-and-color.md
  + #/.claude/rules/typography-and-spacing.md
  + #/.claude/agents/design-critic.md
  + #/examples/apple-demo/index.html
  + #/examples/brandkit-demo/index.html
  + #/examples/brandkit-demo/theme.css
  + #/examples/component-states/app-shell.html
  + #/examples/component-states/button.html
  + #/examples/component-states/card.html
  + #/examples/component-states/charts.html
  + #/examples/component-states/command-palette.html
  + #/examples/component-states/context-menu.html
  + #/examples/component-states/data-display.html
  + #/examples/component-states/data-table.html
  + #/examples/component-states/datepicker.html
  + #/examples/component-states/drawer.html
  + #/examples/component-states/edge-cases.html
  + #/examples/component-states/feedback.html
  + #/examples/component-states/fileupload.html
  + #/examples/component-states/form-controls.html
  + #/examples/component-states/icons.js
  + #/examples/component-states/input.html
  + #/examples/component-states/misc.html
  + #/examples/component-states/modal.html
  + #/examples/component-states/navigation.html
  + #/examples/component-states/overlays.html
  + #/examples/component-states/search.html
  + #/examples/component-states/select.html
  + #/examples/component-states/tabs.html
  + #/examples/component-states/toast.html
  + #/examples/directions.css
  + #/examples/foldable/index.html
  + #/examples/golden/Button.tsx
  + #/examples/golden/Modal.tsx
  + #/examples/golden/README.md
  + #/examples/golden/components.css
  + #/examples/golden/render-harness.css
  + #/examples/golden/theme.css
  + #/examples/index.html
  + #/examples/sample-app/BRIEF.md
  + #/examples/sample-app/Dashboard.tsx
  + #/examples/sample-app/Settings.tsx
  + #/examples/sample-app/StatCard.tsx
  + #/examples/sample-app/preview.html
  + #/examples/sample-app/styles.css
  + #/examples/showcase/index.html
  + #/examples/sites/gov/index.html
  + #/examples/sites/gov/theme.css
  + #/examples/sites/journal/fonts/OFL-Anuphan.txt
  + #/examples/sites/journal/fonts/OFL-NotoSerifThai.txt
  + #/examples/sites/journal/fonts/journal-display.woff2
  + #/examples/sites/journal/fonts/journal-reading.woff2
  + #/examples/sites/journal/index.html
  + #/examples/sites/journal/theme.css
  + #/examples/sites/launch/index.html
  + #/examples/sites/launch/theme.css
  + #/examples/sites/local/index.html
  + #/examples/sites/local/theme.css
  + #/examples/sites/store/index.html
  + #/examples/sites/store/product.html
  + #/examples/sites/store/store.css
  + #/examples/sites/store/store.js
  + #/examples/sites/store/theme.css
  + #/examples/templates/_console.css
  + #/examples/templates/atc.html
  + #/examples/templates/banking.html
  + #/examples/templates/checkout.html
  + #/examples/templates/crm.html
  + #/examples/templates/daw.html
  + #/examples/templates/dispatch.html
  + #/examples/templates/editor.html
  + #/examples/templates/emr.html
  + #/examples/templates/genomics.html
  + #/examples/templates/grid.html
  + #/examples/templates/logistics.html
  + #/examples/templates/mes.html
  + #/examples/templates/orbital.html
  + #/examples/templates/rail.html
  + #/examples/templates/shell.css
  + #/examples/templates/soc.html
  + #/examples/templates/tracing.html
  + #/examples/templates/training.html
  + #/examples/templates/trial.html
  + #/examples/templates/warehouse.html
  + #/examples/templates/weather.html
  + #/examples/terminal/index.html
  + #/examples/thumbs/apple-dark.png
  + #/examples/thumbs/apple-light.png
  + #/examples/thumbs/atlas-dark.png
  + #/examples/thumbs/atlas-light.png
  + #/examples/thumbs/brandkit-dark.png
  + #/examples/thumbs/brandkit-light.png
  + #/examples/thumbs/compare-kit.png
  + #/examples/thumbs/compare-slop.png
  + #/examples/thumbs/reference-dark.png
  + #/examples/thumbs/reference-light.png
  + #/examples/thumbs/site-gov.png
  + #/examples/thumbs/site-journal.png
  + #/examples/thumbs/site-launch.png
  + #/examples/thumbs/site-local.png
  + #/examples/thumbs/site-store.png
  + #/examples/thumbs/terminal-light.png
  + #/examples/thumbs/terminal-thumb.png
  + #/examples/thumbs/tpl-atc.png
  + #/examples/thumbs/tpl-banking.png
  + #/examples/thumbs/tpl-checkout.png
  + #/examples/thumbs/tpl-crm.png
  + #/examples/thumbs/tpl-daw.png
  + #/examples/thumbs/tpl-dispatch.png
  + #/examples/thumbs/tpl-editor.png
  + #/examples/thumbs/tpl-emr.png
  + #/examples/thumbs/tpl-genomics.png
  + #/examples/thumbs/tpl-grid.png
  + #/examples/thumbs/tpl-logistics.png
  + #/examples/thumbs/tpl-mes.png
  + #/examples/thumbs/tpl-orbital.png
  + #/examples/thumbs/tpl-rail.png
  + #/examples/thumbs/tpl-soc.png
  + #/examples/thumbs/tpl-tracing.png
  + #/examples/thumbs/tpl-training.png
  + #/examples/thumbs/tpl-trial.png
  + #/examples/thumbs/tpl-warehouse.png
  + #/examples/thumbs/tpl-weather.png
  + #/templates/product-design/.claude/commands/gate.md
  + #/templates/product-design/.claude/rules/accessibility.md
  + #/templates/product-design/.claude/rules/components.md
  + #/templates/product-design/.claude/rules/taste.md
  + #/templates/product-design/.claude/rules/tokens.md
  + #/templates/product-design/.claude/settings.json
  + #/templates/product-design/.claude/skills/.gitkeep
  + #/templates/product-design/.mcp.json
  + #/templates/product-design/CLAUDE.local.md.template
  + #/templates/product-design/CLAUDE.md
  + #/templates/product-design/design-tokens.json
  + #/templates/product-design/public/images/.gitkeep
  + #/templates/product-design/reference/.gitkeep
  + #/templates/product-design/src/components/.gitkeep

Done. 284 copied, 0 skipped → #
Design systems arrive when used: npx ux-ui-agent-skills@3.1.0 library get <name> (the agentruns scripts/library.mjs itself).
Open the folder in Claude Code: CLAUDE.md loads automatically. Try /design-tokens, /design-code, /apply-aesthetic.
```