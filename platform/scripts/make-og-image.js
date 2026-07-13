/*
 * Generates platform/public/og-image.png (1200x630) with zero dependencies.
 * A dark stage, one glowing microlensing light curve, and the wordmark drawn
 * as a simple 5x7 bitmap font. Run once (or after a rebrand):
 *   node scripts/make-og-image.js
 * Rasterizing in Node without a canvas lib means hand-rolling the PNG via the
 * built-in zlib — same approach ARCHITECTURE.md §4c reserves for per-curve cards.
 */
const fs = require("fs");
const path = require("path");
const zlib = require("zlib");

const W = 1200, H = 630;
const buf = Buffer.alloc(W * H * 4); // RGBA

function setPx(x, y, r, g, b, a = 255) {
  x = Math.round(x); y = Math.round(y);
  if (x < 0 || x >= W || y < 0 || y >= H) return;
  const i = (y * W + x) * 4;
  // simple src-over alpha blend onto existing pixel
  const ia = a / 255, na = 1 - ia;
  buf[i]     = Math.round(r * ia + buf[i] * na);
  buf[i + 1] = Math.round(g * ia + buf[i + 1] * na);
  buf[i + 2] = Math.round(b * ia + buf[i + 2] * na);
  buf[i + 3] = 255;
}

// --- background: vertical gradient of the app's ground tones (#101014 -> #16161c) ---
for (let y = 0; y < H; y++) {
  const t = y / H;
  const r = Math.round(0x10 + (0x16 - 0x10) * t);
  const g = Math.round(0x10 + (0x16 - 0x10) * t);
  const b = Math.round(0x14 + (0x1c - 0x14) * t);
  for (let x = 0; x < W; x++) setPx(x, y, r, g, b);
}

// --- faint graph-paper grid ---
for (let x = 0; x <= W; x += 48) for (let y = 0; y < H; y++) setPx(x, y, 148, 160, 190, 10);
for (let y = 0; y <= H; y += 48) for (let x = 0; x < W; x++) setPx(x, y, 148, 160, 190, 10);

// --- microlensing curve with a soft glow (cyan --cyan #5ab6d8) ---
const N = 1200;
const padX = 90, plotW = W - padX * 2;
const baseY = 430, ampY = 210;
function curveY(t) {
  const bump = Math.exp(-Math.pow((t - 0.5) / 0.05, 2));
  const noise = (Math.sin(t * 140) + Math.sin(t * 61)) * 0.012;
  return baseY - (bump + noise) * ampY;
}
function stroke(width, r, g, b, a) {
  let prev = null;
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    const x = padX + t * plotW, y = curveY(t);
    if (prev) {
      const steps = Math.max(1, Math.round(Math.hypot(x - prev.x, y - prev.y)));
      for (let s = 0; s <= steps; s++) {
        const px = prev.x + (x - prev.x) * (s / steps);
        const py = prev.y + (y - prev.y) * (s / steps);
        for (let dx = -width; dx <= width; dx++)
          for (let dy = -width; dy <= width; dy++)
            if (dx * dx + dy * dy <= width * width) setPx(px + dx, py + dy, r, g, b, a);
      }
    }
    prev = { x, y };
  }
}
stroke(9, 90, 182, 216, 22);  // outer glow
stroke(5, 90, 182, 216, 55);  // mid glow
stroke(2, 190, 230, 245, 255); // core line

// --- wordmark "DISCORD" as a 5x7 bitmap font ---
const FONT = {
  A: ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
  C: ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
  D: ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
  E: ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
  G: ["01111", "10000", "10000", "10111", "10001", "10001", "01110"],
  I: ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
  L: ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
  M: ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
  N: ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
  O: ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
  R: ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
  S: ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
  T: ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
  U: ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
  V: ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
  W: ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
  Z: ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
};
function drawText(text, x0, y0, scale, r, g, b) {
  let cx = x0;
  for (const ch of text) {
    const glyph = FONT[ch];
    if (glyph) {
      for (let ry = 0; ry < 7; ry++)
        for (let rx = 0; rx < 5; rx++)
          if (glyph[ry][rx] === "1")
            for (let sx = 0; sx < scale; sx++)
              for (let sy = 0; sy < scale; sy++)
                setPx(cx + rx * scale + sx, y0 + ry * scale + sy, r, g, b);
      cx += 6 * scale;
    } else {
      cx += 3 * scale;
    }
  }
}
drawText("DISCORD", padX, 90, 11, 242, 242, 245); // wordmark, near-white

// tagline (smaller)
drawText("CITIZEN SCIENCE MICROLENSING REVIEW", padX, 195, 3, 156, 156, 166);

// --- encode PNG ---
function crc32(b) {
  let c = ~0;
  for (let i = 0; i < b.length; i++) {
    c ^= b[i];
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return ~c >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, "ascii");
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
  return Buffer.concat([len, typeBuf, data, crc]);
}
const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4);
ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0; // 8-bit RGBA
// filtered scanlines (filter byte 0 per row)
const raw = Buffer.alloc(H * (1 + W * 4));
for (let y = 0; y < H; y++) {
  raw[y * (1 + W * 4)] = 0;
  buf.copy(raw, y * (1 + W * 4) + 1, y * W * 4, (y + 1) * W * 4);
}
const idat = zlib.deflateSync(raw, { level: 9 });
const png = Buffer.concat([sig, chunk("IHDR", ihdr), chunk("IDAT", idat), chunk("IEND", Buffer.alloc(0))]);

const out = path.join(__dirname, "..", "public", "og-image.png");
fs.writeFileSync(out, png);
console.log(`Wrote ${out} (${png.length} bytes, ${W}x${H})`);
