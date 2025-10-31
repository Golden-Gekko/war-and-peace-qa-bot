import { marked } from 'marked';

marked.setOptions({
    gfm: true,
    breaks: true, // Convert single line breaks into <br>
    mangle: false,
    headerIds: false
});

const chatMessages = document.getElementById('chat-messages');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const typingIndicator = document.getElementById('typing-indicator');

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});
// Send message function
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;
    // Add user message to chat
    addMessageToChat(message, 'user');
    
    // Clear input and reset height
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    // Disable input during processing
    messageInput.disabled = true;
    sendButton.disabled = true;
    
    // Show typing indicator
    chatMessages.scrollTop = chatMessages.scrollHeight;
    try {
        // Create agent message container
        let agentMessageElement = null;
        agentMessageElement = document.createElement('div');
        agentMessageElement.className = 'message agent-message';
        agentMessageElement.innerHTML = `
            <div class="message-header">Агент "Война и Мир"</div>
            <div class="message-content"></div>
        `;
        agentMessageElement.querySelector('.message-content').innerHTML = marked.parse('*Агент в раздумьях...*');
        chatMessages.appendChild(agentMessageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        await new Promise(resolve => requestAnimationFrame(resolve));

        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let agentMessage = '';
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value, { stream: true });
            agentMessage += chunk;
            
            if (agentMessageElement) {
                agentMessageElement.querySelector('.message-content').innerHTML = marked.parse(agentMessage);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }
    } catch (error) {
        console.error('Error:', error);
        addMessageToChat('Извините, произошла ошибка при обработке вашего запроса.', 'agent');
        typingIndicator.style.display = 'none';
    } finally {
        // Re-enable input
        messageInput.disabled = false;
        sendButton.disabled = false;
        messageInput.focus();
    }
}
// Add message to chat
function addMessageToChat(message, sender) {
    const messageElement = document.createElement('div');
    messageElement.className = `message ${sender}-message`;
    
    if (sender === 'user') {
        messageElement.innerHTML = `
            <div class="message-header">Вы</div>
            <div class="message-content">${message}</div>
        `;
    } else {
        messageElement.innerHTML = `
            <div class="message-header">Агент "Война и Мир"</div>
            <div class="message-content">${message}</div>
        `;
    }
    
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
// Event listeners
sendButton.addEventListener('click', sendMessage);
        
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
// Focus input on load
messageInput.focus();