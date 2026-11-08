#!/usr/bin/env node
/* Rebuild webapp/js/assets-data.js from the frozen prompts and reference maps.
   Run from the repository root:  node webapp/tools/build_assets.js */
'use strict';

const fs = require('node:fs');
const path = require('node:path');

const webapp = path.resolve(__dirname, '..');
const root = path.resolve(webapp, '..');

const REF_FILES = {
  global: 'template_land_water_90x180.csv',
  csc: 'template_region_south_china_sea_111x141.csv',
  dzh: 'template_region_mediterranean_71x231.csv',
  seto: 'template_region_seto_inland_sea_64x253.csv',
  arctic01: 'template_region_arctic_nwp_61x241.csv',
  hecate: 'template_region_hecate_strait_127x127.csv'
};

const prompts = {};
const refs = {};
for (const key of Object.keys(REF_FILES)) {
  prompts[key] = fs.readFileSync(path.join(root, 'experiment_config', 'prompts', `${key}.txt`), 'utf8').replace(/^\uFEFF/, '');
  refs[key] = fs.readFileSync(path.join(root, 'quant', 'refs', REF_FILES[key]), 'utf8').replace(/^\uFEFF/, '');
}

const out = `/* Generated from experiment_config/prompts/*.txt and quant/refs/*.csv.
   Kept as plain strings so the page runs from file:// without a server.
   Regenerate with: node webapp/tools/build_assets.js */
window.MAP_ASSETS = {
  prompts: ${JSON.stringify(prompts, null, 2)},
  refs: ${JSON.stringify(refs, null, 2)}
};
`;

fs.writeFileSync(path.join(webapp, 'js', 'assets-data.js'), out, 'utf8');
console.log(`wrote js/assets-data.js (${out.length} bytes)`);