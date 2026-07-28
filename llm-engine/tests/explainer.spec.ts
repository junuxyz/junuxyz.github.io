import { expect, test } from "@playwright/test";

test("navigates the lesson and keeps stable fragments", async ({ page }) => {
  await page.goto("/llm-engine/");
  await expect(
    page.getByRole("heading", { name: "Follow one request" })
  ).toBeVisible();

  await page.keyboard.press("ArrowRight");
  await expect(page).toHaveURL(/#tokenization$/);
  await expect(
    page.getByRole("heading", { name: "Text is not what the model sees" })
  ).toBeVisible();

  await page.keyboard.press("ArrowRight");
  await expect(page).toHaveURL(/#scheduling$/);
});

test("opens directly to a concept fragment", async ({ page }) => {
  await page.goto("/llm-engine/#prefill");
  await expect(page).toHaveURL(/#prefill$/);
  await expect(
    page.getByRole("heading", {
      name: "Prefill moves a whole prompt at once"
    })
  ).toBeVisible();
});

test("lets learners click through the request boundaries", async ({ page }) => {
  await page.goto("/llm-engine/#request");
  await page.getByRole("button", { name: /Scheduler queue/ }).click();
  await expect(
    page.getByText("Admits the request when tokens and KV pages fit.")
  ).toBeVisible();
  await expect(page.getByText("req_7f3a")).toBeVisible();

  await page.getByRole("button", { name: "Next boundary →" }).click();
  await expect(
    page.getByText("Packs ragged requests into model input tensors.")
  ).toBeVisible();
});

test("links text spans to vocabulary tokens", async ({ page }) => {
  await page.goto("/llm-engine/#tokenization");
  await page.getByRole("button", { name: "Rare pieces" }).click();
  await page.getByRole("button", { name: /believ 982/ }).click();
  await expect(page.getByText("characters 2–7")).toBeVisible();
  await expect(page.getByText("vocab id 982")).toBeVisible();
});

test("makes the scheduler budget visible", async ({ page }) => {
  await page.goto("/llm-engine/#scheduling");
  await page.getByLabel("Scheduler token budget").fill("6");
  await page.getByRole("button", { name: "Run iteration →" }).click();
  await expect(page.getByText("4 scheduled")).toBeVisible();
  await expect(page.getByText("6", { exact: true }).last()).toBeVisible();
});

test("steps decode and exposes cache reuse", async ({ page }) => {
  await page.goto("/llm-engine/#decode");
  await page.getByRole("button", { name: "Generate token →" }).click();
  await expect(page.getByText("It", { exact: true }).first()).toBeVisible();
  await page.getByRole("button", { name: "KV position 0, prompt" }).click();
  await expect(page.getByText("reused from prefill")).toBeVisible();
});

test("closes the stream on EOS and releases KV pages", async ({ page }) => {
  await page.goto("/llm-engine/#streaming");
  const nextToken = page.getByRole("button", { name: "Next token →" });
  for (let index = 0; index < 6; index += 1) {
    await nextToken.click();
  }
  await expect(
    page.getByText("stream closed · KV pages released")
  ).toBeVisible();
  await expect(page.getByText("released", { exact: true })).toBeVisible();
});

test("shows how prefix reuse changes physical allocation", async ({ page }) => {
  await page.goto("/llm-engine/#prefix-cache");
  const toggle = page.getByRole("button", { name: "Prefix cache" });
  await expect(toggle).toHaveAttribute("aria-pressed", "false");
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("3 cache hits")).toBeVisible();
  await expect(page.getByText("shared safely")).toBeVisible();
});

test("shows why chunked prefill prevents idle batch slots", async ({ page }) => {
  await page.goto("/llm-engine/#batching");
  const diagram = page.getByRole("group", {
    name: "UNDER LOAD › CONTINUOUS BATCHING"
  });
  await diagram.getByRole("button", { name: "Next iteration →" }).click();
  await expect(diagram.getByText("8 / 8")).toBeVisible();

  await diagram.getByRole("button", { name: "Chunked prefill" }).click();
  await diagram.getByRole("button", { name: "Next iteration →" }).click();
  await expect(diagram.getByText("2 / 8")).toBeVisible();
  await expect(diagram.getByText("idle slots")).toBeVisible();
});

test("verifies speculative tokens instead of trusting the draft", async ({
  page
}) => {
  await page.goto("/llm-engine/#speculative");
  const advance = page
    .getByRole("group", { name: "FASTER DECODE › SPECULATIVE" })
    .getByRole("button", { name: "Advance →" });
  await advance.click();
  await advance.click();
  await advance.click();
  await expect(page.getByText("✓ reuses")).toBeVisible();
  await expect(page.getByText("× values")).toBeVisible();
});

test("keeps all controls usable with reduced motion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/llm-engine/#distributed");
  const diagram = page.getByRole("group", {
    name: "SCALE OUT › TENSOR PARALLELISM"
  });
  await diagram.getByRole("button", { name: "4 GPUs" }).click();
  await diagram.getByRole("button", { name: "Advance →" }).click();
  await expect(page.getByText("weight shard 4/4")).toBeVisible();
});
