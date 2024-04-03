import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ipywidgets import Output

from ipywidgets import HTML, Label, Dropdown, Textarea, Text, Button, Output, VBox, HBox, Layout
from IPython.display import display


class UIHelper:
    def __init__(self, notebook_helper):
        self.nh = notebook_helper

    def create_api_key_input(self):

        label = widgets.Label(
            value='Enter your OpenAI API key if you did not set it in the cell above or in your environment variables, otherwise leave it blank:',
            style={'description_width': 'initial'}
        )
        
        api_key_input = widgets.Text(
            value=self.nh.api_key,
            description='Enter your OpenAI API key (min length 3):',
            disabled=False,
            style={'description_width': 'initial'}
        )
        api_key_button = widgets.Button(description="Save API Key")
        api_key_button.on_click(lambda _: self.nh.save_api_key(api_key_input.value))
        api_key_box = widgets.VBox([api_key_input, api_key_button])
        return api_key_box

    def create_model_dropdown(self):
        model_dropdown = widgets.Dropdown(
            options=self.nh.model_options,
            value=self.nh.model,
            description='OpenAI Model:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        return model_dropdown

    def create_temperature_input(self):
        temperature_input = widgets.FloatSlider(
            value=self.nh.temperature,
            min=0.0,
            max=2.0,
            step=0.1,
            description='Temperature:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            style={'description_width': 'initial'}
        )
        return temperature_input

    def create_filename_input(self):
        filename_input = widgets.Text(
            value=self.nh.filename,
            description='Filename (without extension):',
            disabled=False,
            style={'description_width': 'initial'}
        )
        return filename_input

    def create_total_examples_input(self):
        total_examples_input = widgets.IntText(
            value=self.nh.total_examples,
            description='Total Number of Examples:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        return total_examples_input

    def create_examples_per_request_input(self):
        examples_per_request_input = widgets.IntText(
            value=self.nh.examples_per_request,
            description='Examples per Request:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        return examples_per_request_input

    def create_submit_button(self):
        submit_button = widgets.Button(description="Submit")
        submit_button.on_click(self.on_submit_click)
        return submit_button

    def create_reset_button(self):
        reset_button = widgets.Button(description="Reset")
        reset_button.on_click(self.on_reset_click)

        return reset_button
    
    def on_reset_click(self, _):
        default_values = self.nh.reset_values()

        self.model_dropdown.value = default_values['model']
        self.temperature_input.value = default_values['temperature']
        self.filename_input.value = default_values['filename']
        self.total_examples_input.value = default_values['total_examples']
        self.examples_per_request_input.value = default_values['examples_per_request']

    def on_submit_click(self, button):
        model = self.model_dropdown.value
        temperature = self.temperature_input.value
        filename = self.filename_input.value
        total_examples = self.total_examples_input.value
        examples_per_request = self.examples_per_request_input.value
        self.nh.update_settings(model=model, temperature=temperature, filename=filename,
                                total_examples=total_examples, examples_per_request=examples_per_request)
        self.nh.print_settings()

    def display_ui(self):
        self.api_key_box = self.create_api_key_input()
        self.model_dropdown = self.create_model_dropdown()
        self.temperature_input = self.create_temperature_input()
        self.filename_input = self.create_filename_input()
        self.total_examples_input = self.create_total_examples_input()
        self.examples_per_request_input = self.create_examples_per_request_input()
        self.submit_button = self.create_submit_button()
        self.reset_button = self.create_reset_button()

        display(self.api_key_box)
        display(self.model_dropdown)
        display(self.temperature_input)
        display(self.filename_input)
        display(self.total_examples_input)
        display(self.examples_per_request_input)
        button_box = widgets.HBox([self.submit_button, self.reset_button])
        display(button_box)

    ##############
    ## TEMPLATE ##
    ##############
        
        
    def create_template_dropdown(self):
        """Create a dropdown widget for selecting templates."""
        templates = self.nh.load_templates()
        default_template = 'blank_template'
        return widgets.Dropdown(options=templates, value=default_template if default_template in templates else None)

    def create_template_content_input(self):
        """Create a textarea widget for editing template content."""
        return widgets.Textarea(rows=10)

    def create_filename_input(self):
        """Create a text input widget for entering a new template filename."""
        return widgets.Text(value='new_template')

    def create_save_button(self):
        """Create a button widget for saving the template."""
        button = widgets.Button(description='Save')
        button.on_click(self.save_template)
        return button

    def create_use_button(self):
        """Create a button widget for using the template."""
        button = widgets.Button(description='Use Template')
        button.on_click(self.nh.use_template)
        return button
    
    def save_template(self, _):
        """Save the template content to a file."""
        content = self.template_content_input.value
        filename = self.filename_input.value
        warning = self.nh.save_template(content, filename)
        if warning:
            with self.save_warning_output:
                print(warning)
        else:
            with self.success_output:
                print(f'Template saved as "{filename}.j2"')
            
        self.update_template_dropdown()
    
    def update_template_dropdown(self):
        """Update the template dropdown options."""
        templates = self.nh.load_templates()
        self.template_dropdown.options = templates
        
    def create_template_manager(self):
        display(HTML('''
        <style>
            .widget-label {
                white-space: normal;
                word-wrap: break-word;
                overflow-wrap: break-word;
            }
        </style>
        '''))

        self.template_dropdown = self.create_template_dropdown()
        self.template_content_input = self.create_template_content_input()
        self.filename_input = self.create_filename_input()
        save_button = self.create_save_button()
        use_button = self.create_use_button()
        self.save_warning_output = Output()
        self.success_output = Output()
        self.preview_output = Output()

        input_layout = widgets.VBox([
            widgets.Label(value='Template Manager', style={'font_weight': 'bold', 'font_size': '18px'}),
            widgets.Label(value='Create a new template, or select/modify an existing one.'),
            self.template_dropdown,
            widgets.Label(value='Edit the template content:'),
            self.template_content_input,
            widgets.Label(value="If you have created a new template or modified an existing one, enter a name for it here. Skip this step if you're directly using an existing template without modifying it."),
            self.filename_input,
        ])

        save_button_layout = widgets.HBox([save_button])
        warning_and_success_layout = widgets.VBox([
            self.save_warning_output,
            self.success_output
        ])
        separator_layout = widgets.HBox([widgets.Label(value='─' * 50, style={'font_size': '20px'})])
        use_button_description = widgets.Label(value='''Once you have decided on a template, let's use it. Once you press the button below, you'll be asked to fill in the relevant variables.''',
                                            layout=widgets.Layout(width='auto'))
        use_button_layout = widgets.HBox([use_button])
        button_layout = widgets.VBox([
            save_button_layout,
            warning_and_success_layout,
            separator_layout,
            use_button_description,
            use_button_layout
        ], layout=widgets.Layout(margin='20px 0px'))

        output_layout = widgets.VBox([
            self.preview_output,
        ])

        main_layout = widgets.VBox([input_layout, button_layout, output_layout], layout=widgets.Layout(width='auto'))
        display(main_layout)

        def on_template_selected(change):
            template_content = self.nh.load_template_content(change)
            self.template_content_input.value = template_content

        self.template_dropdown.observe(on_template_selected, names='value')

    def create_template_form(self, variables, template_name):
        """Create a form for filling in template variables."""
        self.template_name = template_name
        load_form = VBox()
        self.placeholders = {}
        for var in variables:
            placeholder_input = Text(
                description=f'Enter value for "{var}":',
                layout=Layout(width='auto', min_width='200px'),
                style={'description_width': 'initial'}
            )
            load_form.children += (placeholder_input,)
            self.placeholders[var] = placeholder_input
        self.output_filename_input = Text(
            description='Enter a filename to save the template with these values (e.g. my_prompt):',
            layout=Layout(width='auto', min_width='200px'),
            style={'description_width': 'initial'})
        load_form.children += (self.output_filename_input,)
        render_button = Button(description='Save and Render')
        self.render_warning_output = Output()
        warning_and_buttons_layout = VBox([
            render_button,
            self.render_warning_output
        ])
        load_form.children += (warning_and_buttons_layout,)
        # Apply CSS styling to the form container
        load_form.layout.width = '100%'
        load_form.layout.min_width = '400px'
        load_form.add_class('my-form')
        display(load_form)
        render_button.on_click(self.nh.on_render_and_save)