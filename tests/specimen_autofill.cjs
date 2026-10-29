// Execute the real UI handlers against a minimal DOM; no server/database writes.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor(tag = 'div') { this.tagName = tag.toUpperCase(); this.children = []; this.value = ''; this.hidden = false; }
  get options() { return this.children; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  addEventListener() {}
  dispatchEvent() {}
}
const elements = new Map();
const selectIds = ['fabric-category', 'material', 'technique', 'pattern', 'texture', 'drapability', 'weight', 'source', 'specimen', 'analysis-mode'];
function get(id) {
  if (!elements.has(id)) elements.set(id, new Element(selectIds.includes(id) ? 'select' : 'div'));
  return elements.get(id);
}
let pending = new Map();
const context = vm.createContext({document: {getElementById: get, createElement: tag => new Element(tag)},
  fetch: url => {
    if (!url.startsWith('/api/specimens/')) return new Promise(() => {});
    return new Promise(resolve => pending.set(url.split('/').pop(), record => resolve({ok: true, json: async () => record})));
  }, Event: class {}});
vm.runInContext(fs.readFileSync(require('node:path').join(__dirname, '../frontend/app.js'), 'utf8'), context);
const sample = (category, properties = {}) => ({name: category + ' sample', reviewer: 'reviewer', sha256: 'hash',
  fabric: {fabric_category: category, material: category, technique: 'unknown', pattern_type: 'solid', fabric_texture: 'unknown', properties}});
async function run() {
  get('fabric-category').value = 'cotton'; get('source').value = 'user_measured'; get('reference').value = 'original manual note';
  get('gsm').value = '125';
  get('specimen').value = 'first'; const first = get('specimen').onchange();
  assert.equal(get('manual-fabric').disabled, true);
  pending.get('first')(sample('leather', {
    drapability: {value: 'structured', source: 'expert_reviewed', reference: 'expert A'},
    gsm: {value: 300, source: 'supplier_documented', reference: 'sheet B'},
    stretch_percent: {value: 0, source: 'user_measured', reference: 'test C'}}));
  await first;
  assert.equal(get('fabric-category').value, 'leather');
  assert.equal(get('gsm').value, 300); assert.equal(get('stretch').value, 0);
  assert.equal(get('source').value, 'unknown'); assert.equal(get('reference').value, '');
  assert.equal(get('manual-provenance').hidden, true);
  assert.match(get('saved-provenance').children.map(el => el.textContent).join(' '), /expert A.*sheet B.*test C/);
  vm.runInContext('resetSuggestions()', context);
  assert.equal(get('fabric-category').value, 'leather', 'photo reset must not alter saved values');
  get('specimen').value = 'old'; const old = get('specimen').onchange();
  get('specimen').value = 'new'; const newer = get('specimen').onchange();
  pending.get('new')(sample('linen')); await newer;
  pending.get('old')(sample('nylon')); await old;
  assert.equal(get('fabric-category').value, 'linen', 'late response must not replace selected sample');
  get('specimen').value = ''; await get('specimen').onchange();
  assert.equal(get('fabric-category').value, 'cotton'); assert.equal(get('gsm').value, '125');
  assert.equal(get('reference').value, 'original manual note');
  assert.equal(get('manual-fabric').disabled, false); assert.equal(get('saved-provenance').hidden, true);
  get('specimen').value = 'late'; const late = get('specimen').onchange();
  get('specimen').value = ''; await get('specimen').onchange();
  pending.get('late')(sample('wool')); await late;
  assert.equal(get('fabric-category').value, 'cotton');
  vm.runInContext(`typicalProfiles = [{id: 'test_preset', name: 'Test profile', source_url: 'https://example.com/spec',
    fabric: {fabric_category: 'denim', properties: {gsm: {value: 432, source: 'typical_profile', reference: 'TEST ONLY'}}}}]`, context);
  get('typical-profile').value = 'test_preset'; get('use-typical').onclick();
  assert.equal(get('fabric-category').value, 'denim'); assert.equal(get('gsm').value, 432);
  assert.equal(get('manual-fabric').disabled, true); assert.equal(get('save-specimen').disabled, true);
  assert.equal(vm.runInContext('manualFabric().properties.gsm.source', context), 'typical_profile');
  get('clear-typical').onclick();
  assert.equal(get('gsm').value, '125'); assert.equal(get('reference').value, 'original manual note');
  assert.equal(get('manual-fabric').disabled, false);
  get('use-typical').onclick(); vm.runInContext('resetSuggestions()', context);
  assert.equal(vm.runInContext('activeTypical', context), null);
  assert.equal(get('manual-fabric').disabled, false);
  console.log('Saved autofill, zero stretch, per-property provenance, draft restore, photo reset and stale response checks passed.');
}
run().catch(error => { console.error(error); process.exitCode = 1; });
