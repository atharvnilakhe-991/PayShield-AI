const input = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const preview = document.getElementById('preview');
const previewWrap = document.getElementById('previewWrap');
const analyzeButton = document.getElementById('analyzeButton');
let selectedFile = null;
let objectUrl = null;

function state(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }
function resetResult() { ['loadingState','resultState','errorState'].forEach(hide); state('emptyState'); }
function choose(file) {
  if (!file) return;
  const allowed = ['image/png','image/jpeg','image/webp'];
  if (!allowed.includes(file.type) || file.size > 15 * 1024 * 1024) {
    selectedFile = null; analyzeButton.disabled = true; previewWrap.classList.add('hidden'); dropzone.classList.remove('hidden');
    resetResult(); hide('emptyState'); state('errorState');
    document.getElementById('errorState').textContent = 'Choose a PNG, JPEG, or WebP image under 15 MB.';
    return;
  }
  selectedFile = file;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  preview.src = objectUrl;
  document.getElementById('fileName').textContent = file.name;
  dropzone.classList.add('hidden'); previewWrap.classList.remove('hidden'); analyzeButton.disabled = false;
  resetResult();
}

input.addEventListener('change', () => choose(input.files?.[0]));
document.getElementById('clearButton').addEventListener('click', () => {
  selectedFile = null; input.value = ''; analyzeButton.disabled = true;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null; previewWrap.classList.add('hidden'); dropzone.classList.remove('hidden'); resetResult();
});
['dragenter','dragover'].forEach(name => dropzone.addEventListener(name, e => { e.preventDefault(); dropzone.classList.add('dragging'); }));
['dragleave','drop'].forEach(name => dropzone.addEventListener(name, e => { e.preventDefault(); dropzone.classList.remove('dragging'); }));
dropzone.addEventListener('drop', e => choose(e.dataTransfer.files?.[0]));
analyzeButton.addEventListener('click', async () => {
  if (!selectedFile) return;
  hide('emptyState'); hide('resultState'); hide('errorState'); state('loadingState'); analyzeButton.disabled = true;
  const body = new FormData(); body.append('image', selectedFile);
  try {
    const response = await fetch('/api/analyze', {method:'POST', body});
    const result = await response.json();
    hide('loadingState');
    if (!response.ok) throw new Error(result.error || 'Could not analyze this image.');
    document.getElementById('guess').textContent = result.model_guess === 'edited' ? 'Model flagged possible edits' : 'Model found no strong edit signal';
    document.getElementById('score').textContent = `${(result.edited_score_uncalibrated * 100).toFixed(1)}%`;
    document.getElementById('scoreBar').style.width = `${Math.max(0,Math.min(100,result.edited_score_uncalibrated*100))}%`;
    document.getElementById('scoreNote').textContent = `Uncalibrated score; not a verified probability. Displayed guess uses the validation-selected threshold ${(result.decision_threshold_validation_selected*100).toFixed(1)}%.`;
    document.getElementById('localizationNote').textContent = result.localization_note;
    document.getElementById('warningText').textContent = result.warning;
    document.getElementById('dimensions').textContent = `Input: ${result.image_width} × ${result.image_height} px · score is uncalibrated`;
    state('resultState');
  } catch (error) {
    hide('loadingState'); state('errorState');
    document.getElementById('errorState').textContent = error.message || 'Analysis could not be completed. Check the image and try again.';
  } finally { analyzeButton.disabled = !selectedFile; }
});
