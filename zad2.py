import gradio as gr

def get_file_info(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    num_lines = len(lines)
    text_lines = read_text_file(file_name)
    info = f"Plik zawiera {num_lines} wierszy."
    classes = {}
    for line in text_lines:
        class_name = line.split()[0]
        if class_name not in classes:
            classes[class_name] = 0
        classes[class_name] += 1
    return classes, info

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def display_file(file_name, num_lines_to_display):
    classes, info = get_file_info(file_name)
    response = "Liczba klas decyzyjnych to: {}\n".format(len(classes))
    for class_name, class_size in classes.items():
        response += "Wielkość klasy {} to: {}\n".format(class_name, class_size)

    lines = read_text_file(file_name)
    num_lines_to_display = int(num_lines_to_display)
    displayed_lines = "\n".join(lines[:num_lines_to_display])
    return f"{info}\n{response}\nWybrane wiersze:\n{displayed_lines}"

file_name_input = gr.inputs.Textbox(label="Podaj nazwę pliku:")
num_lines_input = gr.inputs.Number(label="Wybierz liczbę wierszy do wyświetlenia:", default=1)
output_text = gr.outputs.Textbox(label="Wynik:")

iface = gr.Interface(fn=display_file, inputs=[file_name_input, num_lines_input], outputs=output_text, title="CHAT BOT",
                     description="Podaj nazwę pliku oraz liczbę wierszy do wyświetlenia, aby uzyskać wynik.")
iface.launch()