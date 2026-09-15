from dotenv import load_dotenv
import os
from openai import AzureOpenAI, OpenAI
import instructor

load_dotenv()

# Native OpenAI is the default and Azure the fallback.
DEFAULT_PROVIDER = "openai"
FALLBACK_PROVIDER = "azure"

# Both providers return 429 under bulk load; the SDK backs off and retries.
DEFAULT_MAX_RETRIES = 8


def resolve_provider(provider=None):
    """Pick the provider, preferring native OpenAI when it is usable."""
    choice = (provider or os.getenv("DATAFORAGER_LLM_PROVIDER") or "").lower()
    if choice in ("openai", "azure"):
        return choice
    if os.getenv("OPENAI_API_KEY"):
        return DEFAULT_PROVIDER
    return FALLBACK_PROVIDER


class OpenAIClient:
    def __init__(self, provider=None):
        # The underlying client is created lazily on first use (see the `client`
        # property) so that importing modules which hold a module-level
        # OpenAIClient() does not require credentials to be set.
        self._client = None
        self._provider = provider
        self.text_generation_model_default = "gpt-4o-mini"
        self.embedding_model_default = "text-embedding-3-small"

    @property
    def provider(self):
        """Which endpoint this client talks to, resolved on first access."""
        return resolve_provider(self._provider)

    @property
    def client(self):
        """The OpenAI client, instantiated on first access."""
        if self._client is None:
            if self.provider == "openai":
                self._client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    max_retries=DEFAULT_MAX_RETRIES,
                )
            else:
                self._client = AzureOpenAI(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version="2024-10-01-preview",
                    max_retries=DEFAULT_MAX_RETRIES,
                )
        return self._client

    def infer_metadata(self, messages, response_model, model=None, temperature=0.1,
                       return_usage=False):
        if model is None:
            model = self.text_generation_model_default
        try:
            client = instructor.from_openai(self.client)
            if not return_usage:
                return client.chat.completions.create(
                    model=model,
                    response_model=response_model,
                    messages=messages,
                    temperature=temperature
                )
            response, completion = client.chat.completions.create_with_completion(
                model=model,
                response_model=response_model,
                messages=messages,
                temperature=temperature
            )
            usage = getattr(completion, "usage", None)
            return response, {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            }
        except Exception as e:
            print(f"Error inferring metadata: {e}")
            return (None, {"prompt_tokens": 0, "completion_tokens": 0}) if return_usage else None

    def infer_metadata_wo_instructor(self, messages, response_format=None, model=None):
        if model is None:
            model = self.text_generation_model_default
        try:
            response = self.client.chat.completions.create(
                model=model,
                response_format=response_format,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error inferring metadata: {e}")
            return

    def generate_embeddings(self, text, model=None):
        if model is None:
            model = self.embedding_model_default
        try:
            text = text.replace("\n", " ")
            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return
    
    # Azure OpenAI Assistants tutorial: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/assistant
    def create_assistant(self, name, instructions, model=None):
        if model is None:
            model = self.text_generation_model_default
        try:
            assistant = self.client.beta.assistants.create(
                name=name,
                instructions=instructions,
                tools=[{"type": "code_interpreter"}],  # in case user input sql query
                model=model
            )
            return assistant
        except Exception as e:
            print(f"Error creating assistant: {e}")
            return
    
    # Thread is essentially the record of the conversation session between the assistant and the user
    def create_thread(self):
        try:
            thread = self.client.beta.threads.create()
            return thread
        except Exception as e:
            print(f"Error creating thread: {e}")
            return
    
    def create_message(self, thread_id, role, content):
        try:
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role=role,
                content=content
            )
            return message
        except Exception as e:
            print(f"Error creating message: {e}")
            return
    
    def list_thread_messages(self, thread_id):
        try:
            thread_messages = self.client.beta.threads.messages.list(
                thread_id=thread_id
            )
            return thread_messages
        except Exception as e:
            print(f"Error listing thread message: {e}")
            return
    
    def run_thread(self, thread_id, assistant_id, instructions=None):
        try:
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                instructions=instructions  # optional new instructions, will override the default instructions provided in `create_assistant`
            )
            return run
        except Exception as e:
            print(f"Error running thread: {e}")
            return

    def run_status(self, thread_id, run_id):
        try:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            return run.status
        except Exception as e:
            print(f"Error monitoring run status: {e}")
            return