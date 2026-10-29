const $ = selector => document.querySelector(selector);
let catalogue = [];

async function loadGarments() {
  const response = await fetch('/api/garments');
  catalogue = (await response.json()).garments;
  populateGarments();
}

function populateGarments() {
  const matching = catalogue.filter(g => g.family === $('#family').value);
  $('#garment').innerHTML = matching.map(g => `<option value="${g.id}">${g.name}</option>`).join('');
}

$('#family').addEventListener('change', populateGarments);
$('#tryon-form').addEventListener('submit', async event => {
  event.preventDefault();
  const photo = $('#photo').files[0];
  if (!photo) return;
  const response = await fetch('/api/try-on', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({photoName: photo.name, garmentId: $('#garment').value})});
  const result = await response.json();
  if (!response.ok) { $('#summary').textContent = result.error; return; }
  const imageUrl = URL.createObjectURL(photo);
  const preview = $('#preview');
  preview.innerHTML = `<img src="${imageUrl}" alt="Uploaded person portrait"><i class="overlay" style="--colour:${result.garment.colour}"></i>`;
  $('#badge').textContent = result.route === 'traditional_draping' ? 'Traditional draping branch' : 'Global try-on branch';
  $('#summary').innerHTML = `<b>${result.garment.name}</b> routed through <b>${result.route.replace('_', ' ')}</b>.<br>${result.checks.join(' · ')}.<br><em>${result.notice}</em>`;
});

loadGarments().catch(() => { $('#summary').textContent = 'Could not connect to the Component 1 API. Start backend/app.py and reload.'; });
