//API
const API_BASE = 'http://localhost:5000';

document.addEventListener('DOMContentLoaded', () => {
  //Analysis page
  const fileInput  = document.getElementById('fileInput');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const preview    = document.getElementById('preview');
  const resultDiv  = document.getElementById('result');

  if (fileInput && analyzeBtn && preview && resultDiv) {
    //Enable analyze button & preview thumbnail
    fileInput.addEventListener('change', () => {
      analyzeBtn.disabled = !fileInput.files.length;
      if (fileInput.files.length) {
        const reader = new FileReader();
        reader.onloadend = () => {
          preview.src           = reader.result;
          preview.style.display = 'block';
        };
        reader.readAsDataURL(fileInput.files[0]);
      } else {
        preview.style.display = 'none';
      }
    });

    //Upload and call /predict
    analyzeBtn.addEventListener('click', async () => {
      analyzeBtn.textContent = 'Analysing…';
      analyzeBtn.disabled   = true;
      resultDiv.innerHTML   = '';

      const form = new FormData();
      form.append('image', fileInput.files[0]);

      try {
        const res  = await fetch(`${API_BASE}/predict`, {
          method: 'POST',
          body: form
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || res.statusText);

        //Build result HTML
        let html = `
          <p><strong>Normal:</strong>   ${(data.normal    * 100).toFixed(1)} %</p>
          <p><strong>Benign:</strong>   ${(data.benign    * 100).toFixed(1)} %</p>
          <p><strong>Malignant:</strong>${(data.malignant * 100).toFixed(1)} %</p>
          <hr>
          <p><strong>Final prediction:</strong> ${data.predicted}
             (${(data.confidence * 100).toFixed(1)} %)</p>
        `;

        //Recommendation based on class & confidence
        if (data.predicted === 'malignant' && data.malignant > 0.5) {
          html += `<p class="alert alert-danger"><strong>Recommendation:</strong> High malignancy probability (>50%). Please schedule biopsy ASAP.</p>`;
        } else if (data.predicted === 'benign' && data.benign > 0.5) {
          html += `<p class="alert alert-warning"><strong>Recommendation:</strong> Likely benign (>50%). Follow up with your doctor.</p>`;
        } else if (data.predicted === 'normal' && data.normal > 0.5) {
          html += `<p class="alert alert-success"><strong>Recommendation:</strong> Normal image (>50%). No further action needed.</p>`;
        } else {
          html += `<p class="alert"><strong>Recommendation:</strong> Moderate confidence. Consider specialist consultation.</p>`;
        }

        resultDiv.innerHTML = html;

      } catch (err) {
        console.error(err);
        resultDiv.innerHTML = `<p class="error">${err.message}</p>`;
      } finally {
        analyzeBtn.textContent = 'Analyze Image';
        analyzeBtn.disabled    = false;
      }
    });
  }

  //Registration page


  const registerForm = document.getElementById('registerForm');
  const registerMsg  = document.getElementById('registerMsg');
  
  if (registerForm) {
    registerForm.onsubmit = async (e) => {
      e.preventDefault();
      registerMsg.textContent = '';
      registerMsg.style.color = 'black';
  
      const formData = new FormData(registerForm);
      const data = {
        username:        formData.get('username'),
        confirmUsername: formData.get('confirmUsername'),
        email:           formData.get('email'),
        password:        formData.get('password'),
        confirmPassword: formData.get('confirmPassword')
      };
  
      //Client-side validation
      if (data.username !== data.confirmUsername) {
        registerMsg.textContent = 'Usernames do not match.';
        registerMsg.style.color = 'red';
        return;
      }
  
      if (data.password !== data.confirmPassword) {
        registerMsg.textContent = 'Passwords do not match.';
        registerMsg.style.color = 'red';
        return;
      }
  
      try {
        const res = await fetch(`${API_BASE}/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });
  
        const result = await res.json();
        if (!res.ok) throw new Error(result.error || res.statusText);
  
        registerMsg.textContent = result.message || 'Registration successful!';
        registerMsg.style.color = 'green';
  
        //Optional: redirect to login
        //setTimeout(() => window.location.href = '/login.html', 1500);
  
      } catch (err) {
        registerMsg.textContent = err.message;
        registerMsg.style.color = 'red';
      }
    };
  }
  

  //Login
  document.getElementById('loginForm').onsubmit = async function (e) {
    e.preventDefault();
  
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value.trim();
  
    const res = await fetch('http://localhost:5000/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
  
    const data = await res.json();
  
    if (res.ok) {
      alert(data.message);
      window.location.href = 'analysis.html';
    } else {
      alert(data.error);
    }
  };
  

  //Logout

const logoutBtn = document.getElementById('logoutBtn');

if (window.location.pathname.includes('analysis.html')) {
  const user = localStorage.getItem('user');
  if (!user) {
    alert('Primero inicia sesión.');
    window.location.href = 'login.html';
  }
}
});
