import { existsSync, readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const MANIFEST_PATH = join(
  ROOT,
  'apps',
  'web',
  'public',
  'replay',
  'ninapro',
  'emg-replay.json',
);
const HAS_REST_WARNING = Boolean(existsSync(MANIFEST_PATH) && (
  JSON.parse(readFileSync(MANIFEST_PATH, 'utf8')) as {
    signalQuality?: {
      restCalibration?: {
        flaggedChannels?: number[];
      };
    };
  }
).signalQuality?.restCalibration?.flaggedChannels?.length);

test('Live reports high rest RMS as a non-blocking calibration warning', async ({ page }) => {
  test.skip(!HAS_REST_WARNING, 'No flagged local NinaPro rest calibration is staged.');
  await page.goto('/');
  await page.getByLabel('Replay source').selectOption('ninapro');

  const warning = page.getByRole('status').filter({
    hasText: 'Recorded rest RMS screen flagged channel',
  });
  await expect(warning).toContainText('above the exploratory 50 µV RMS screen');
  await expect(warning).toContainText('Replay continues');
  await expect(warning).toContainText('not a validated fault diagnosis');
});
