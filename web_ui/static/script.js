document.getElementById("send-btn").addEventListener("click", sendMessage);
document.getElementById("user-input").addEventListener("keypress", function (e) {
  if (e.key === "Enter") sendMessage();
});

async function sendMessage() {
  const input = document.getElementById("user-input");
  const text = input.value.trim();
  if (!text) return;

  appendMessage("user", text);
  input.value = "";

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    const data = await res.json();
    appendMessage("bot", data.response);
  } catch (err) {
    appendMessage("bot", "Error: Unable to reach model.");
  }
}

function appendMessage(sender, message) {
  const chatBox = document.getElementById("chat-box");
  const msgDiv = document.createElement("div");
  msgDiv.classList.add("chat-message");
  msgDiv.classList.add(sender === "user" ? "user-message" : "bot-message");
  msgDiv.textContent = message;
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}
