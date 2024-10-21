import openvino_genai
import argparse
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

# Initialize the rich console
console = Console()

def streamer(subword, output_buffer, live_panel):
    """Stream tokens directly to the buffer and update the live view."""
    output_buffer.append(subword)
    markdown_content = Markdown("".join(output_buffer))
    # Update the live markdown content
    live_panel.update(markdown_content)
    return False

def get_multiline_input():
    """Get multiline input from user until they press Ctrl+D (Unix) or Ctrl+Z (Windows)."""
    console.print("[bold cyan]Enter your question (Press Ctrl+D on Unix/Linux or Ctrl+Z on Windows when done):[/bold cyan]")
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass  # User pressed Ctrl+D/Ctrl+Z
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    args = parser.parse_args()
    
    device = 'CPU'  # Set the device to GPU or CPU
    pipe = openvino_genai.LLMPipeline(args.model_dir, device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 1024
    config.temperature = 0.1

    pipe.start_chat()
    print(f"Model Compiled to {device}. Now you can start the chat.")

    while True:
        # Input the prompt
        prompt = get_multiline_input()
        
        if prompt == '/bye':
            break
        
        console.clear()

        # Display user input with distinct formatting
        user_input_text = Text("User Input:\n", style="bold yellow")
        user_input_text.append(f"{prompt}\n", style="cyan")
        console.print(user_input_text)

        # Separator to visually distinguish input and output
        console.print("[bold green]--- Generating response ---[/bold green]\n")

        # Prepare an output buffer to accumulate tokens
        output_buffer = []

        # Use Live to dynamically update the model's response during output generation
        with Live("", refresh_per_second=15) as live_panel:
            pipe.generate(prompt, config, lambda subword: streamer(subword, output_buffer, live_panel))
        
        # Print a divider after the response is completed
        console.print("\n[bold green]--- End of response ---[/bold green]\n")
    
    pipe.finish_chat()

if '__main__' == __name__:
    main()
