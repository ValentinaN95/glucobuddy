import os
import requests
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load environment variables
load_dotenv()

class LLMGlucoseAssistant:
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct"):
        """Initialize LLM for glucose data analysis and conversation"""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        """Load the LLM model and tokenizer"""
        try:
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.pipeline = None
    
    def create_glucose_context(self, glucose_data, user_question=""):
        """Create context from glucose data for the LLM"""
        if not glucose_data or 'egvs' not in glucose_data:
            return "No glucose data available."
        
        readings = glucose_data['egvs']
        if not readings:
            return "No glucose readings found."
        
        # Analyze data
        df = pd.DataFrame(readings)
        df['systemTime'] = pd.to_datetime(df['systemTime'])
        df['value'] = pd.to_numeric(df['value'])
        
        # Calculate key metrics
        avg_glucose = df['value'].mean()
        min_glucose = df['value'].min()
        max_glucose = df['value'].max()
        std_glucose = df['value'].std()
        
        # Time in range calculations
        in_range = df[(df['value'] >= 70) & (df['value'] <= 180)]
        time_in_range = len(in_range) / len(df) * 100
        hypo_count = len(df[df['value'] < 70])
        hyper_count = len(df[df['value'] > 180])
        
        # Recent trends (last 24 hours)
        recent_data = df[df['systemTime'] >= (datetime.now() - timedelta(hours=24))]
        if len(recent_data) > 0:
            recent_avg = recent_data['value'].mean()
            recent_trend = "stable"
            if len(recent_data) > 1:
                if recent_data['value'].iloc[-1] > recent_data['value'].iloc[0] + 20:
                    recent_trend = "increasing"
                elif recent_data['value'].iloc[-1] < recent_data['value'].iloc[0] - 20:
                    recent_trend = "decreasing"
        else:
            recent_avg = avg_glucose
            recent_trend = "no recent data"
        
        context = f"""
        GLUCOSE DATA SUMMARY (Last 7 days):
        - Average glucose: {avg_glucose:.1f} mg/dL
        - Range: {min_glucose:.0f} - {max_glucose:.0f} mg/dL
        - Standard deviation: {std_glucose:.1f} mg/dL
        - Time in range (70-180 mg/dL): {time_in_range:.1f}%
        - Hypoglycemic episodes (<70 mg/dL): {hypo_count}
        - Hyperglycemic episodes (>180 mg/dL): {hyper_count}
        - Recent 24h average: {recent_avg:.1f} mg/dL
        - Recent trend: {recent_trend}
        - Total readings: {len(df)}
        """
        
        return context
    
    def chat_with_glucose_data(self, user_question, glucose_data, conversation_history=""):
        """Generate intelligent response using LLM with glucose context"""
        if not self.pipeline:
            return "Sorry, the AI assistant is not available. Please check the model loading."
        
        glucose_context = self.create_glucose_context(glucose_data, user_question)
        
        system_prompt = """You are GlucoBuddy, an AI assistant specialized in diabetes management and glucose analysis. 
        You are friendly, supportive, and knowledgeable about diabetes care. Always provide helpful, accurate information while 
        reminding users to consult with their healthcare providers for medical decisions.

        Guidelines:
        - Be encouraging and supportive
        - Provide practical, actionable advice
        - Explain complex concepts in simple terms
        - Always emphasize the importance of medical supervision
        - Use emojis occasionally to be friendly
        - Focus on education and empowerment"""
        
        prompt = f"""<|system|>
{system_prompt}

CURRENT GLUCOSE DATA:
{glucose_context}

{conversation_history}
<|end|>
<|user|>
{user_question}
<|end|>
<|assistant|>
"""
        
        try:
            response = self.pipeline(
                prompt,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract only the assistant's response
            full_text = response[0]['generated_text']
            assistant_response = full_text.split('<|assistant|>')[-1].strip()
            
            return assistant_response
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}. Please try again."

class DexcomAPI:
    def __init__(self):
        self.client_id = os.getenv('DEXCOM_CLIENT_ID')
        self.client_secret = os.getenv('DEXCOM_CLIENT_SECRET')
        self.redirect_uri = os.getenv('DEXCOM_REDIRECT_URI')
        self.base_url = os.getenv('DEXCOM_SANDBOX_BASE_URL')
        self.access_token = None
        self.refresh_token = None
    
    def get_authorization_url(self):
        """Generate OAuth2 authorization URL"""
        auth_url = f"{self.base_url}/v2/oauth2/login"
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': 'offline_access'
        }
        return f"{auth_url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
    
    def exchange_code_for_token(self, authorization_code):
        """Exchange authorization code for access token"""
        token_url = f"{self.base_url}/v2/oauth2/token"
        
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': authorization_code,
            'grant_type': 'authorization_code',
            'redirect_uri': self.redirect_uri
        }
        
        response = requests.post(token_url, data=data)
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data['access_token']
            self.refresh_token = token_data.get('refresh_token')
            return True
        return False
    
    def refresh_access_token(self):
        """Refresh access token using refresh token"""
        if not self.refresh_token:
            return False
            
        token_url = f"{self.base_url}/v2/oauth2/token"
        
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'refresh_token': self.refresh_token,
            'grant_type': 'refresh_token'
        }
        
        response = requests.post(token_url, data=data)
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data['access_token']
            return True
        return False
    
    def get_glucose_readings(self, start_date=None, end_date=None):
        """Retrieve glucose readings"""
        if not self.access_token:
            return None
            
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
            
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        params = {
            'startDate': start_date.isoformat(),
            'endDate': end_date.isoformat()
        }
        
        response = requests.get(
            f"{self.base_url}/v3/users/self/egvs",
            headers=headers,
            params=params
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            if self.refresh_access_token():
                return self.get_glucose_readings(start_date, end_date)
        
        return None

class EnhancedGlucoBuddy:
    def __init__(self):
        self.dexcom_api = DexcomAPI()
        self.llm_assistant = LLMGlucoseAssistant()
        self.conversation_history = []
        self.current_glucose_data = None
    
    def authenticate_user(self, auth_code):
        """Authenticate user with Dexcom"""
        if self.dexcom_api.exchange_code_for_token(auth_code):
            return "✅ Authentication successful! GlucoBuddy is now connected to your Dexcom data."
        else:
            return "❌ Authentication error. Please verify the code."
    
    def load_glucose_data(self):
        """Load latest glucose data"""
        self.current_glucose_data = self.dexcom_api.get_glucose_readings()
        return self.current_glucose_data is not None
    
    def chat_with_glucobuddy(self, user_message, history):
        """Enhanced chat interface with LLM"""
        if not self.current_glucose_data:
            if not self.load_glucose_data():
                return history + [("I need access to your glucose data first. Please authenticate with Dexcom.", "")]
        
        # Get LLM response
        conversation_context = "\n".join([f"User: {h[0]}\nGlucoBuddy: {h[1]}" for h in history[-3:]])  # Last 3 exchanges
        
        response = self.llm_assistant.chat_with_glucose_data(
            user_message, 
            self.current_glucose_data,
            conversation_context
        )
        
        # Update history
        history.append((user_message, response))
        return history, ""
    
    def create_glucose_chart(self):
        """Create glucose chart"""
        if not self.current_glucose_data or 'egvs' not in self.current_glucose_data:
            return None
        
        readings = self.current_glucose_data['egvs']
        if not readings:
            return None
        
        df = pd.DataFrame(readings)
        df['systemTime'] = pd.to_datetime(df['systemTime'])
        df['value'] = pd.to_numeric(df['value'])
        df = df.sort_values('systemTime')
        
        fig = go.Figure()
        
        # Main glucose line
        fig.add_trace(go.Scatter(
            x=df['systemTime'],
            y=df['value'],
            mode='lines+markers',
            name='Glucose',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Target zones
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="Hypoglycemia threshold (70 mg/dL)")
        fig.add_hline(y=180, line_dash="dash", line_color="orange",
                     annotation_text="Hyperglycemia threshold (180 mg/dL)")
        
        # Target area
        fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.1,
                     annotation_text="Target range", annotation_position="top left")
        
        fig.update_layout(
            title="📈 GlucoBuddy - Your Glucose Trend (Last 7 days)",
            xaxis_title="Date and Time",
            yaxis_title="Glucose (mg/dL)",
            height=500,
            showlegend=True
        )
        
        return fig

# Initialize Enhanced GlucoBuddy
buddy = EnhancedGlucoBuddy()

def create_gradio_interface():
    with gr.Blocks(title="Enhanced GlucoBuddy - AI Glucose Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 Enhanced GlucoBuddy - Your AI Glucose Assistant")
        gr.Markdown("Hi! I'm your enhanced GlucoBuddy with AI conversation capabilities. Ask me anything about your glucose data!")
        
        with gr.Tab("🔐 Authentication"):
            gr.Markdown("### Step 1: Dexcom Authorization")
            auth_url = buddy.dexcom_api.get_authorization_url()
            gr.Markdown(f"[Click here to authorize GlucoBuddy]({auth_url})")
            
            gr.Markdown("### Step 2: Enter authorization code")
            auth_code_input = gr.Textbox(
                label="Authorization code",
                placeholder="Enter the code from Dexcom..."
            )
            auth_button = gr.Button("Authenticate", variant="primary")
            auth_status = gr.Textbox(label="Status", interactive=False)
            
            auth_button.click(
                fn=buddy.authenticate_user,
                inputs=[auth_code_input],
                outputs=[auth_status]
            )
        
        with gr.Tab("💬 Chat with GlucoBuddy"):
            gr.Markdown("### Ask me anything about your glucose data!")
            
            chatbot = gr.Chatbot(
                height=500,
                placeholder="Hello! I'm your AI GlucoBuddy. Ask me about your glucose patterns, trends, or any diabetes-related questions!"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your question here... (e.g., 'How has my glucose been trending?', 'Why am I having highs at night?')",
                    container=False,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            # Example questions
            gr.Markdown("### 💡 Try asking me:")
            example_questions = [
                "How is my time in range?",
                "What's causing my morning highs?",
                "Am I having too many lows?",
                "How can I improve my glucose stability?",
                "What patterns do you see in my data?"
            ]
            
            for question in example_questions:
                gr.Button(question, variant="secondary").click(
                    fn=lambda q=question: buddy.chat_with_glucobuddy(q, []),
                    outputs=[chatbot, msg]
                )
            
            def respond(message, history):
                return buddy.chat_with_glucobuddy(message, history)
            
            msg.submit(respond, [msg, chatbot], [chatbot, msg])
            send_btn.click(respond, [msg, chatbot], [chatbot, msg])
        
        with gr.Tab("📊 Glucose Chart"):
            gr.Markdown("### Your Glucose Visualization")
            chart_button = gr.Button("📈 Show My Chart", variant="primary")
            chart_output = gr.Plot()
            
            chart_button.click(
                fn=buddy.create_glucose_chart,
                outputs=[chart_output]
            )
        
        with gr.Tab("ℹ️ About Enhanced GlucoBuddy"):
            gr.Markdown("""
            ### 🧠 What's New in Enhanced GlucoBuddy?
            
            This enhanced version includes:
            
            - **🤖 AI Conversation**: Chat naturally about your glucose data
            - **📊 Intelligent Analysis**: Ask specific questions and get personalized answers
            - **💡 Smart Insights**: The AI understands patterns and provides contextual advice
            - **🔄 Interactive Learning**: Continuous conversation that builds on previous exchanges
            
            ### 🚀 How to Use:
            
            1. **Authenticate** with your Dexcom account
            2. **Chat** with GlucoBuddy about your glucose data
            3. **Ask specific questions** like "Why are my mornings high?" or "How can I improve?"
            4. **Get personalized advice** based on your actual data patterns
            
            ### 🛡️ Privacy & Safety:
            - All processing happens locally
            - No data is stored permanently
            - AI provides educational information only
            - Always consult your healthcare provider for medical decisions
            
            **Enhanced GlucoBuddy - Making diabetes management more intelligent and interactive!**
            """)
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )