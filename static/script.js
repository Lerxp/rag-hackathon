// Client-side logic for querying /ask and rendering either
// a full JSON answer or a streamed text response controlled by .env.
document.addEventListener("DOMContentLoaded", () => {
  // Containers for dynamic content and a simple loading indicator
  const answerEl = document.getElementById("answer");
  const matchesEl = document.getElementById("matches");
  const spinner = document.getElementById("spinner");

  // Render top matches into a simple <ul> list
  function renderMatches(matches) {
    if (!matches || !matches.length) {
      matchesEl.innerHTML = "";
      return;
    }
    matchesEl.innerHTML =
      `<h3>Top Matches</h3><ul>` +
      matches.map(m => `<li>${m.source} p.${m.page} — score ${m.score}</li>`).join("") +
      `</ul>`;
  }

  // Consume ReadableStream from fetch and append decoded text chunks
  async function streamTextResponse(res) {
    // Hide spinner once streaming begins
    spinner.style.display = "none";

    try {
      const header = res.headers.get('X-Matches');
      if (header) {
        const matches = JSON.parse(header);
        renderMatches(matches);
      }
    } catch (e) {
      console.warn('Failed to parse X-Matches header:', e);
    }

    answerEl.innerHTML = `<h2>Answer</h2><pre id="answer-text" style="white-space: pre-wrap"></pre>`;
    const target = document.getElementById('answer-text');

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      target.textContent += decoder.decode(value, { stream: true });
    }
  }

  // Submit handler: decides between JSON and streaming based on Content-Type
  document.getElementById("ask-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = document.getElementById("query").value;
    answerEl.textContent = "";
    matchesEl.textContent = "";

    // Show spinner when request starts
    spinner.style.display = "inline-block";

    try {
      const res = await fetch(`/ask?query=${encodeURIComponent(q)}`);
      const contentType = res.headers.get('content-type') || '';

      if (contentType.includes('application/json')) {
        spinner.style.display = "none";
        const data = await res.json();
        answerEl.innerHTML = `<h2>Answer</h2><p>${data.answer}</p>`;
        renderMatches(data.matches || []);
      } else if (contentType.includes('text/plain')) {
        await streamTextResponse(res);
      } else {
        spinner.style.display = "none";
        answerEl.innerHTML = `<p>Unexpected response type: ${contentType}</p>`;
      }
    } catch (err) {
      spinner.style.display = "none";
      answerEl.innerHTML = `<p>Error: ${err.message}</p>`;
    }
  });
});
