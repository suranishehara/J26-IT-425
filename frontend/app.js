const $ = id => document.getElementById(id);
function node(tag, text, cls) { const el = document.createElement(tag); el.textContent = text; if (cls) el.className = cls; return el; }
function options(id, values) { for (const value of values) { const el = node('option', value.replaceAll('_', ' ')); el.value = value; $(id).append(el); } }
async function request(url, data) {
  const response = await fetch(url, data ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) } : {});
  const result = await response.json(); if (!response.ok) throw Error(result.error || 'Request failed'); return result;
}
let fields = [], photoVersion = 0, inputVersion = 0;
const labelControls = {fabric_category: 'fabric-category', material: 'material', technique: 'technique', pattern_type: 'pattern', fabric_texture: 'texture'};
const propertyControls = {drapability: 'drapability', weight_band: 'weight', gsm: 'gsm', stretch_percent: 'stretch'};
let manualDraft = null, specimenRequestVersion = 0;
let typicalProfiles = [], activeTypical = null, typicalDraft = null;
function clearTypical() {
  if (!activeTypical) return;
  activeTypical = null;
  for (const [id, value] of Object.entries(typicalDraft || {})) setRecordValue(id, value);
  typicalDraft = null;
  $('manual-fabric').disabled = false; $('manual-provenance').hidden = false;
  $('save-specimen').disabled = false; $('clear-typical').hidden = true;
  $('typical-status').textContent = '';
}
request('/api/typical-profiles').then(data => {
  typicalProfiles = data.profiles;
  for (const profile of typicalProfiles) {
    const option = node('option', `${profile.name} (${profile.category})`); option.value = profile.id; $('typical-profile').append(option);
  }
}).catch(error => { $('typical-status').textContent = 'Typical profiles unavailable: ' + error.message; });
$('use-typical').onclick = () => {
  const profile = typicalProfiles.find(item => item.id === $('typical-profile').value);
  if (!profile || $('specimen').value) { $('typical-status').textContent = 'Choose manual mode and a profile variant first.'; return; }
  if (!activeTypical) typicalDraft = captureManualDraft();
  activeTypical = profile; showFabricRecord(profile.fabric);
  $('manual-fabric').disabled = true; $('manual-provenance').hidden = true;
  $('save-specimen').disabled = true; $('clear-typical').hidden = false;
  $('optional-measurements').open = true;
  $('typical-status').replaceChildren(node('span', `Typical profile · ${profile.name} · `));
  const source = node('a', 'Source'); source.href = profile.source_url; source.target = '_blank'; source.rel = 'noopener noreferrer'; $('typical-status').append(source);
  $('use-typical').dispatchEvent(new Event('input', {bubbles:true}));
};
$('clear-typical').onclick = () => { clearTypical(); $('clear-typical').dispatchEvent(new Event('input', {bubbles:true})); };
function setRecordValue(id, value) {
  const control = $(id);
  if (control.tagName === 'SELECT' && !Array.from(control.options).some(option => option.value === value)) options(id, [value]);
  control.value = value;
}
function showFabricRecord(fabric) {
  for (const [field, id] of Object.entries(labelControls)) setRecordValue(id, fabric[field] || 'unknown');
  for (const [field, id] of Object.entries(propertyControls)) {
    $(id).value = fabric.properties?.[field]?.value ?? (id === 'gsm' || id === 'stretch' ? '' : 'unknown');
  }
  // Saved records may have different sources per property. Never collapse them
  // into the manual form's one shared source/reference.
  $('source').value = 'unknown'; $('reference').value = '';
}
function captureManualDraft() {
  return Object.fromEntries([...Object.values(labelControls), ...Object.values(propertyControls), 'source', 'reference'].map(id => [id, $(id).value]));
}
function manualFabric() {
  if (activeTypical) return {...activeTypical.fabric, reviewed: $('reviewed').checked};
  const properties = {};
  for (const [field, id] of [['drapability', 'drapability'], ['weight_band', 'weight'], ['gsm', 'gsm'], ['stretch_percent', 'stretch']]) {
    properties[field] = {value: $(id).value, source: $('source').value, reference: $('reference').value};
  }
  return {fabric_category: $('fabric-category').value, material: $('material').value, technique: $('technique').value,
    pattern_type: $('pattern').value, fabric_texture: $('texture').value, properties, reviewed: $('reviewed').checked};
}
async function refreshSpecimens(selected = '', guard = () => true) {
  const data = await request('/api/specimens');
  if (!guard()) return false;
  $('specimen').replaceChildren();
  const manual = node('option', 'Manual properties below'); manual.value = ''; $('specimen').append(manual);
  for (const record of data.specimens) {
    const option = node('option', `${record.name} · reviewer ${record.reviewer} · ${record.id.slice(0, 8)}`);
    option.value = record.id; $('specimen').append(option);
  }
  $('specimen').value = selected;
  return true;
}
$('export-catalogue').onclick = async () => {
  try {
    const catalogue = await request('/api/catalogue/export');
    const url = URL.createObjectURL(new Blob([JSON.stringify(catalogue, null, 2)], {type: 'application/json'}));
    const link = document.createElement('a'); link.href = url; link.download = 'fabric-properties-catalogue.json';
    document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    $('catalogue-status').textContent = `Exported ${catalogue.records.length} sample records.`;
  } catch (error) { $('catalogue-status').textContent = error.message; }
};
$('catalogue-file').onchange = () => { $('catalogue-reviewed').checked = false; };
$('import-catalogue').onclick = async () => {
  $('import-catalogue').disabled = true;
  try {
    const file = $('catalogue-file').files[0];
    if (!file || file.size > 500000) throw Error('Choose a catalogue JSON file below 500 KB.');
    if (!$('catalogue-reviewed').checked) throw Error('Confirm that you reviewed the sample sources.');
    const catalogue = JSON.parse(await file.text());
    const result = await request('/api/catalogue/import', {catalogue, reviewed: true});
    const selected = $('specimen').value;
    const selectionVersion = specimenRequestVersion;
    await refreshSpecimens(selected, () => selectionVersion === specimenRequestVersion); $('specimen').disabled = false;
    $('catalogue-status').textContent = `Imported ${result.imported_count}; skipped ${result.skipped_count} existing/duplicate records. Select a sample above to use it.`;
    $('catalogue-reviewed').checked = false;
  } catch (error) { $('catalogue-status').textContent = error.message; }
  finally { $('import-catalogue').disabled = false; }
};
$('specimen').onchange = async () => {
  clearTypical();
  const id = $('specimen').value;
  const version = ++specimenRequestVersion;
  inputVersion++; clearResults(); $('reviewed').checked = false;
  $('fabric-check').replaceChildren();
  if (id && manualDraft === null) manualDraft = captureManualDraft();
  $('manual-fabric').disabled = Boolean(id);
  $('save-specimen').disabled = Boolean(id);
  $('manual-provenance').hidden = Boolean(id);
  $('saved-provenance').hidden = true; $('saved-provenance').replaceChildren();
  $('specimen-status').textContent = id ? 'Loading recorded sample details…' : 'Manual mode. Your previous draft is restored.';
  if (!id && manualDraft) {
    for (const [control, value] of Object.entries(manualDraft)) setRecordValue(control, value);
    manualDraft = null;
  }
  if (id) {
    showFabricRecord({});
    try {
      const record = await request('/api/specimens/' + encodeURIComponent(id));
      if ($('specimen').value !== id || version !== specimenRequestVersion) return;
      showFabricRecord(record.fabric);
      $('optional-labels').open = Object.keys(labelControls).some(key => key !== 'fabric_category' && record.fabric[key] && record.fabric[key] !== 'unknown');
      $('optional-measurements').open = ['gsm', 'stretch_percent'].some(key => record.fabric.properties[key]);
      $('specimen-status').textContent = `Filled from ${record.name} · reviewer ${record.reviewer}. Read-only; select Manual properties below to restore your draft. This sample is not automatically matched to the uploaded photo.`;
      const provenance = $('saved-provenance'); provenance.hidden = false;
      provenance.append(node('h3', 'Saved property sources'));
      for (const [key, prop] of Object.entries(record.fabric.properties)) provenance.append(node('p', `${key.replaceAll('_', ' ')}: ${prop.value} · ${prop.source.replaceAll('_', ' ')} · ${prop.reference}`));
      provenance.append(node('p', `Record SHA-256: ${record.sha256}`, 'hint'));
    } catch (error) { if ($('specimen').value === id && version === specimenRequestVersion) $('specimen-status').textContent = 'Sample could not be loaded: ' + error.message + ' Select manual mode or retry another sample.'; }
  }
};
$('save-specimen').onclick = async () => {
  $('save-specimen').disabled = true;
  const version = inputVersion;
  try {
    const record = await request('/api/specimens', {name: $('specimen-name').value, reviewer: $('specimen-reviewer').value, fabric: manualFabric()});
    if (version !== inputVersion) {
      $('specimen-status').textContent = 'Record saved, but inputs changed; reload the page to list it. Current draft retained.';
      return;
    }
    if (!await refreshSpecimens(record.id, () => version === inputVersion)) {
      $('specimen-status').textContent = 'Sample saved. Inputs changed during refresh; current draft retained. Refresh to list the saved sample.';
      return;
    }
    await $('specimen').onchange();
    $('specimen').dispatchEvent(new Event('input', {bubbles:true}));
    $('specimen-status').textContent = `Saved ${record.name}. SHA-256 ${record.sha256}. ${record.notice}`;
  } catch (error) { $('specimen-status').textContent = error.message; }
  finally { $('save-specimen').disabled = Boolean($('specimen').value); }
};
$('recommend').disabled = true;
$('check-fabric').onclick = async () => {
  const version = inputVersion;
  $('check-fabric').disabled = true;
  try {
    let fabric = manualFabric();
    const specimenId = $('specimen').value;
    if (specimenId) fabric = (await request('/api/specimens/' + encodeURIComponent(specimenId))).fabric;
    const result = await request('/api/fabric-record/check', activeTypical ? {typical_profile_id: activeTypical.id, reviewed: true} : {fabric});
    if (version !== inputVersion) return;
    const area = $('fabric-check'); area.replaceChildren(node('h3', 'Fabric record summary'));
    for (const key of ['fabric_category', 'material', 'technique', 'pattern_type', 'fabric_texture']) {
      area.append(node('p', `${key.replaceAll('_', ' ')}: ${result.fabric[key].replaceAll('_', ' ')}`));
    }
    for (const [name, property] of Object.entries(result.fabric.properties)) {
      area.append(node('p', `${name.replaceAll('_', ' ')}: ${property.value} · ${property.source.replaceAll('_', ' ')} · ${property.reference}`));
    }
    area.append(node('p', result.ranking_properties.length ? 'Used for design ranking: ' + result.ranking_properties.join(', ') : 'No physical ranking properties recorded; designs will be preference-based.'));
    if (result.context_only_properties.length) area.append(node('p', 'Recorded as context only: ' + result.context_only_properties.join(', ')));
    area.append(node('p', activeTypical ? 'Typical profile · used for this demonstration, not saved as a documented sample.' : (result.can_save_specimen ? 'This record can be saved after review with a specimen name and reviewer ID.' : 'Add at least one documented physical property to save a specimen. You can still continue with unknown properties.')));
    area.append(node('p', result.notice, 'hint'));
  } catch (error) { if (version === inputVersion) $('fabric-check').replaceChildren(node('p', error.message, 'conflict')); }
  finally { $('check-fabric').disabled = false; }
};
request('/api/options').then(async data => {
  options('material', data.materials); options('technique', data.techniques);
  options('fabric-category', data.fabric_categories);
  selectedModelStatus();
  options('pattern', ['unknown', ...data.pattern_type]); options('texture', ['unknown', ...data.fabric_texture]); options('style', data.styles);
  for (const d of data.designs) { const el = node('option', d.name); el.value = d.id; $('silhouette').append(el); }
  fields = data.measurement_fields;
  for (const name of fields) {
    const label = node('label', name.replaceAll('_', ' ')), input = document.createElement('input');
    input.type = 'number'; input.step = 'any'; input.min = '0.01'; input.id = 'measure-' + name; label.append(input); $('measurements').append(label);
  }
  $('recommend').disabled = false;
  try {
    await refreshSpecimens();
    $('specimen').disabled = false;
    $('specimen-status').textContent = $('specimen').options.length > 1 ? 'Choose your documented sample to fill its details, or use manual mode.' : 'No saved samples yet. Enter known properties once, then save a sample for reuse.';
  } catch (error) { $('specimen-status').textContent = 'Saved samples unavailable: ' + error.message + ' Manual entry remains available.'; }
}).catch(error => { $('status').textContent = 'Setup unavailable: ' + error.message; });
let classifierChoices = [];
function selectedModelStatus() {
  const choice = classifierChoices.find(item => item.id === $('analysis-mode').value);
  $('model-status').textContent = choice?.available ? choice.label + ` · ${choice.categories.length} supported categories.` : (choice?.notice || 'Select a classifier and analyse your photograph.');
  if (choice?.available && !$('specimen').value && !activeTypical) {
    const previous = $('fabric-category').value;
    $('fabric-category').replaceChildren();
    options('fabric-category', ['unknown', ...choice.categories]);
    $('fabric-category').value = choice.categories.includes(previous) ? previous : 'unknown';
  }
}
request('/api/health').then(data => {
  classifierChoices = data.classifiers || [];
  for (const option of $('analysis-mode').options) {
    const choice = classifierChoices.find(item => item.id === option.value);
    option.disabled = !choice?.available;
  }
  selectedModelStatus();
}).catch(error => { $('model-status').textContent = 'Classifier availability check failed. ' + error.message; });
function resetSuggestions() { clearTypical(); $('suggestions').replaceChildren(); if (!$('specimen').value) $('fabric-category').value = 'unknown'; }
$('photo').onchange = () => { photoVersion++; resetSuggestions(); $('preview').hidden = true; $('photo-status').textContent = 'Photograph changed. Analyse again or enter reviewed labels manually.'; $('reviewed').checked = false; };
$('analyse').onclick = async () => {
  const file = $('photo').files[0], version = photoVersion;
  try {
    if (!file) throw Error('Choose a fabric photograph.');
    if (!$('permission').checked) throw Error('Confirm image permission.');
    if (file.size > 6000000) throw Error('Use an image below 6 MB.');
    $('analyse').disabled = true; resetSuggestions(); $('reviewed').checked = false;
    inputVersion++; clearResults();
    $('photo-status').textContent = 'Analysing locally. First experimental use loads the model; no image is sent to a remote generation API.';
    const image = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = () => reject(Error('Cannot read image')); reader.readAsDataURL(file); });
    if (version !== photoVersion || !$('permission').checked) return;
    const result = await request('/api/analyse-fabric', {image, permission: true, mode: 'experimental', model_id: $('analysis-mode').value});
    if (version !== photoVersion) return;
    $('preview').src = result.preview; $('preview').hidden = false;
    const successfulPrediction = result.classifier?.available && (result.predictions.fabric_category || []).length > 0;
    $('photo-status').textContent = successfulPrediction ? 'Analysis complete. Review the category suggestions below.' : result.notice;
    if (result.capture_quality) {
      for (const warning of result.capture_quality.warnings) $('suggestions').append(node('p', warning, 'warning'));
    }
    const review = result.review_decision;
    if (review && review.status !== 'no_prediction') {
      if (!review.allow_quick_apply) $('suggestions').append(node('h3', 'Uncertain · keep category unknown'));
      if (!review.allow_quick_apply) for (const reason of review.reasons) $('suggestions').append(node('p', reason, 'warning'));
    }
    const suggestions = result.predictions.fabric_category || [];
    if (suggestions.length) {
      $('suggestions').append(node('h3', 'Experimental category suggestions'));
      const list = document.createElement('ol');
      for (const item of suggestions) list.append(node('li', `${item.category.replaceAll('_', ' ')} · relative score ${(item.relative_score * 100).toFixed(1)}%`));
      $('suggestions').append(list);
      const apply = node('button', 'Use top category'); apply.type = 'button';
      apply.disabled = !review || !review.allow_quick_apply || Boolean($('specimen').value) || Boolean(activeTypical);
      apply.onclick = () => {
        if (version !== photoVersion || !$('permission').checked || !review?.allow_quick_apply || $('specimen').value || activeTypical) return;
        $('fabric-category').value = suggestions[0].category;
        $('fabric-category').dispatchEvent(new Event('input', {bubbles:true}));
      };
      $('suggestions').append(apply);
    }
  } catch (error) { if (version === photoVersion) $('photo-status').textContent = error.message; }
  finally { $('analyse').disabled = false; }
};
function clearResults() { $('results').replaceChildren(); $('warnings').replaceChildren(); }
$('workspace').addEventListener('input', event => {
  inputVersion++;
  $('fabric-check').replaceChildren(node('p', 'Record changed. Check again to refresh the summary.', 'hint'));
  if (event.target.id === 'analysis-mode' || event.target.id === 'permission') {
    clearTypical();
    if (event.target.id === 'analysis-mode') selectedModelStatus();
    photoVersion++; resetSuggestions(); $('photo-status').textContent = 'Mode or permission changed. Analyse again or use manual review.';
    if (!$('permission').checked) { $('preview').hidden = true; $('preview').removeAttribute('src'); }
  }
  if (event.target.id !== 'reviewed') $('reviewed').checked = false;
  clearResults(); $('status').textContent = 'Inputs changed. Review and generate updated suggestions.';
});
$('workspace').onsubmit = async event => {
  event.preventDefault(); $('recommend').disabled = true; clearResults();
  const submittedVersion = inputVersion;
  try {
    const properties = {};
    for (const [field, id] of [['drapability', 'drapability'], ['weight_band', 'weight'], ['gsm', 'gsm'], ['stretch_percent', 'stretch']]) {
      properties[field] = {value: $(id).value, source: $('source').value, reference: $('reference').value};
    }
    const measurements = Object.fromEntries(fields.map(name => [name, $('measure-' + name).value]));
    const payload = {fabric: {fabric_category: $('fabric-category').value, material: $('material').value, technique: $('technique').value, pattern_type: $('pattern').value,
      fabric_texture: $('texture').value, properties, reviewed: $('reviewed').checked},
      profile: {unit: $('unit').value, measurements, fit_preference: $('fit').value}, style: $('style').value, silhouette: $('silhouette').value};
    if ($('specimen').value) {
      payload.specimen_id = $('specimen').value; payload.reviewed = $('reviewed').checked; delete payload.fabric;
    }
    if (activeTypical) {
      payload.typical_profile_id = activeTypical.id; payload.reviewed = $('reviewed').checked; delete payload.fabric;
    }
    const result = await request($('compare-properties').checked ? '/api/designs/compare' : '/api/designs', payload);
    if (submittedVersion !== inputVersion) return;
    if (result.property_basis === 'typical_profile') $('warnings').append(node('p', 'Typical profile', 'hint'));
    if (result.rule_config) $('rule-status').textContent = `Rules: ${result.rule_config.version} · review: ${result.rule_config.review.status} · SHA-256 ${result.rule_config.sha256}`;
    for (const warning of result.warnings) $('warnings').append(node('p', warning, 'warning'));
    if (result.comparison) {
      const comparison = result.comparison;
      const details = document.createElement('details'); details.open = true;
      details.append(node('summary', 'Fabric-property impact comparison'));
      details.append(node('p', comparison.notice));
      details.append(node('p', comparison.properties_used_for_ranking.length ? 'Ranking uses: ' + comparison.properties_used_for_ranking.join(', ') : 'No supported ranking properties recorded: both runs should be identical.'));
      details.append(node('p', 'Ranking changed: ' + (comparison.ranking_changed ? 'yes' : 'no') + '. Top design changed: ' + (comparison.top_design_changed ? 'yes' : 'no') + '.'));
      if (comparison.context_only_properties.length) details.append(node('p', 'Recorded but not scored: ' + comparison.context_only_properties.join(', ')));
      const table = document.createElement('table'), head = document.createElement('thead'), header = document.createElement('tr');
      for (const title of ['Design', 'Baseline rank / points', 'Fabric-aware rank / points']) { const th = node('th', title); th.scope = 'col'; header.append(th); }
      head.append(header); table.append(head);
      const body = document.createElement('tbody');
      for (const row of comparison.rows) {
        const tr = document.createElement('tr');
        tr.append(node('td', row.name), node('td', `${row.baseline_rank} / ${row.baseline_score}`), node('td', `${row.property_aware_rank} / ${row.property_aware_score}`)); body.append(tr);
      }
      table.append(body); details.append(table); $('results').append(details);
    }
    if (result.specimen) {
      $('warnings').append(node('p', `Saved specimen: ${result.specimen.name} · ${result.specimen.id} · reviewer ${result.specimen.reviewer} · SHA-256 ${result.specimen.sha256}. Sources are user assertions, not independently verified.`));
      for (const [key, prop] of Object.entries(result.fabric.properties)) $('warnings').append(node('p', `${key}: ${prop.value} · ${prop.source} · ${prop.reference}`));
    }
    for (const design of result.designs) {
      const card = node('article', '', 'design'); card.append(node('h3', design.name), node('p', 'Rule score ' + design.rule_score + ' · provisional / not fit verified'));
      const list = document.createElement('ul');
      for (const reason of design.reasons) list.append(node('li', reason));
      for (const conflict of design.conflicts) list.append(node('li', conflict, 'conflict'));
      card.append(list);
      for (const [name, values] of Object.entries(design.measurements)) {
        card.append(node('p', `${name}: body ${values.body_cm} cm + illustrative ease ${values.illustrative_ease_cm.join('–')} cm → draft circumference ${values.draft_finished_circumference_cm.join('–')} cm (not a cutting pattern).`));
      }
      if (design.missing_measurements.length) card.append(node('p', 'Missing for this design: ' + design.missing_measurements.join(', ')));
      $('results').append(card);
    }
    $('status').textContent = result.notice;
  } catch (error) { if (submittedVersion === inputVersion) $('status').textContent = error.message; }
  finally { $('recommend').disabled = false; }
};
