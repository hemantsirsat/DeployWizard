import typer
from pathlib import Path
from typing import Optional, List, Any
from rich.console import Console
from rich.table import Table
from datetime import datetime

from deploywizard.scaffolder import Scaffolder
from deploywizard import __version__

# Create console instance
console = Console()

# Create the main CLI app
app = typer.Typer(
    help="DeployWizard - ML Model Deployment Tool",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]}
)

def print_version():
    """Print the current version and exit."""
    console.print(f"DeployWizard v{__version__}", style="bold green")
    raise typer.Exit()

# Add version command
@app.command("version", help="Show version and exit.")
def version():
    """Show version and exit."""
    print_version()

# Common options
model_name_option = typer.Option(..., "--name", "-n", help="Name of the model")
version_option = typer.Option(None, "--version", "-v", help="Version of the model (default: latest)")
framework_option = typer.Option(..., "--framework", "-f", help="Model framework (sklearn, pytorch, tensorflow)")
description_option = typer.Option("", "--description", "-d", help="Description of the model")
model_class_option = typer.Option(None, "--model-class", help="Path to Python file containing model class definition (required for PyTorch state_dict)")

# Add version callback to the main app
@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=lambda value: print_version() if value else None,
        is_eager=True
    )
) -> None:
    """DeployWizard - ML Model Deployment Tool."""
    pass

@app.command()
def register(
    model_path: str = typer.Argument(..., help="Path to the model file"),
    name: str = model_name_option,
    version: str = typer.Option("1.0.0", "--version", "-v", help="Version of the model"),
    framework: str = framework_option,
    description: str = description_option,
):
    """Register a new model in the registry."""
    try:
        scaffolder = Scaffolder()
        model_info = scaffolder.register_model(
            name=name,
            version=version,
            model_path=model_path,
            framework=framework,
            description=description
        )
        console.print(f"Successfully registered [bold]{name}[/bold] v{version}", style="green")
        _print_model_info(model_info)
    except Exception as e:
        console.print(f"Error: {str(e)}", style="red")
        raise typer.Exit(code=1)

@app.command(name="list")
def list_models():
    """List all registered models."""
    try:
        scaffolder = Scaffolder()
        models = scaffolder.list_models()
        
        if not models:
            console.print("No models found in the registry.", style="yellow")
            return
            
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Name")
        table.add_column("Version")
        table.add_column("Framework")
        table.add_column("Description")
        table.add_column("Registered At")
        
        for model in models:
            table.add_row(
                model['name'],
                model['version'],
                model['framework'],
                model.get('description', '')[:50] + ('...' if len(model.get('description', '')) > 50 else ''),
                model.get('created_at', '').split('T')[0]
            )
            
        console.print(table)
    except Exception as e:
        console.print(f"Error: {str(e)}", style="red")
        raise typer.Exit(code=1)

@app.command()
def info(
    name: str = model_name_option,
    version: str = version_option,
):
    """Show detailed information about a registered model."""
    try:
        scaffolder = Scaffolder()
        model_info = scaffolder.get_model_info(name, version)
        
        if not model_info:
            version_msg = f"version '{version}'" if version else "latest version"
            console.print(f"Model '{name}' ({version_msg}) not found in registry.", style="red")
            raise typer.Exit(code=1)
            
        _print_model_info(model_info)
    except Exception as e:
        console.print(f"Error: {str(e)}", style="red")
        raise typer.Exit(code=1)

@app.command()
def delete(
    name: str = model_name_option,
    version: str = typer.Option(None, "--version", "-v", help="Version to delete (delete all versions if not specified)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
):
    """Delete a model or model version from the registry."""
    try:
        scaffolder = Scaffolder()
        
        # Get model info for better error messages
        model_info = scaffolder.get_model_info(name, version)
        if not model_info:
            version_msg = f"version '{version}' " if version else ""
            console.print(f"Model '{name}' {version_msg}not found in registry.", style="red")
            raise typer.Exit(code=1)
        
        # Confirm deletion
        if not force:
            if version:
                confirm_msg = f"Are you sure you want to delete {name} v{version}?"
            else:
                confirm_msg = f"Are you sure you want to delete ALL versions of {name}?"
            
            if not typer.confirm(confirm_msg):
                console.print("Operation cancelled.")
                return
        
        # Delete the model
        success = scaffolder._registry.delete_model(name, version)
        
        if success:
            if version:
                console.print(f"Successfully deleted {name} v{version}", style="green")
            else:
                console.print(f"Successfully deleted all versions of {name}", style="green")
        else:
            console.print(f"Failed to delete {name}", style="red")
            raise typer.Exit(code=1)
            
    except Exception as e:
        console.print(f"Error: {str(e)}", style="red")
        raise typer.Exit(code=1)

@app.command()
def update(
    name: str = model_name_option,
    version: str = version_option,
    new_name: str = typer.Option(None, "--new-name", help="New name for the model"),
    new_version: str = typer.Option(None, "--new-version", help="New version for the model"),
    description: str = typer.Option(None, "--description", "-d", help="New description for the model"),
):
    """Update metadata for a registered model."""
    try:
        scaffolder = Scaffolder()
        
        # Get current model info
        model_info = scaffolder.get_model_info(name, version)
        if not model_info:
            version_msg = f" (version: {version}) " if version else " "
            console.print(f"Model '{name}'{version_msg}not found in registry.", style="red")
            raise typer.Exit(code=1)
        
        # Update fields if provided
        updated = False
        
        if new_name and new_name != name:
            # Check if new name already exists
            existing = scaffolder.get_model_info(new_name, model_info['version'])
            if existing and existing['id'] != model_info['id']:
                console.print(f"A model with name '{new_name}' and version {model_info['version']} already exists.", style="red")
                raise typer.Exit(code=1)
            
            # Update name in the registry
            del scaffolder._registry._registry['models'][name][model_info['version']]
            if not scaffolder._registry._registry['models'][name]:
                del scaffolder._registry._registry['models'][name]
                
            model_info['name'] = new_name
            updated = True
        
        if new_version and new_version != model_info['version']:
            # Check if version already exists for this model
            existing = scaffolder.get_model_info(model_info['name'], new_version)
            if existing:
                console.print(f"Version {new_version} already exists for model '{model_info['name']}'", style="red")
                raise typer.Exit(code=1)
                
            # Remove old version and add new one
            del scaffolder._registry._registry['models'][model_info['name']][model_info['version']]
            model_info['version'] = new_version
            updated = True
        
        if description is not None and description != model_info.get('description'):
            model_info['description'] = description
            updated = True
        
        if not updated:
            console.print("No changes detected. Use --help to see available options.")
            return
        
        # Save the updated model
        if 'models' not in scaffolder._registry._registry:
            scaffolder._registry._registry['models'] = {}
        if model_info['name'] not in scaffolder._registry._registry['models']:
            scaffolder._registry._registry['models'][model_info['name']] = {}
            
        scaffolder._registry._registry['models'][model_info['name']][model_info['version']] = model_info
        scaffolder._registry._save_registry()
        
        console.print(f"Successfully updated {model_info['name']} v{model_info['version']}", style="green")
        _print_model_info(model_info)
        
    except Exception as e:
        console.print(f"Error: {str(e)}", style="red")
        raise typer.Exit(code=1)

@app.command()
def deploy(
    name: str = model_name_option,
    version: str = version_option,
    output_dir: str = typer.Option(".", "--output", "-o", help="Output directory"),
    api: str = typer.Option("fastapi", help="Type of API to generate"),
    model_class: str = model_class_option,
):
    """Generate a deployment project for a registered model.
    
    If the model is a PyTorch state_dict, you must provide --model-class pointing to a Python file
    containing the model class definition.
    """
    try:
        scaffolder = Scaffolder()
        console.print(f"Deploying [bold]{name}[/bold] (version: {version or 'latest'})...")
        
        # Get model info to check framework
        model_info = scaffolder.get_model_info(name, version)
        if not model_info:
            version_msg = f"version '{version}'" if version else "latest version"
            console.print(f"Model '{name}' ({version_msg}) not found in registry.", style="red")
            raise typer.Exit(code=1)
            
        # Validate model class is provided for PyTorch state_dict
        if model_info['framework'] == 'pytorch' and model_class:
            model_class_path = Path(model_class)
            if not model_class_path.exists():
                console.print(f"[yellow]Warning:[/yellow] Model class file not found: {model_class}")
                if not typer.confirm("Continue without model class? (may cause errors if model is a state_dict)"):
                    raise typer.Exit()
            else:
                console.print(f"Using model class from: {model_class}")
        
        scaffolder.generate_project(
            model_name=name,
            version=version,
            output_dir=output_dir,
            api_type=api,
            model_class_path=model_class,
        )
        
        console.print(f"Successfully deployed [bold]{name}[/bold] to {output_dir}", style="green")
        console.print("\nNext steps:")
        console.print(f"1. cd {output_dir}")
        console.print("2. docker-compose up --build")
        console.print("\nYour API will be available at http://localhost:8000")
        console.print("API documentation: http://localhost:8000/docs")
        
    except Exception as e:
        console.print(f"Error: {str(e)}", style="red")
        raise typer.Exit(code=1)

@app.command()
def init(
    model: str = typer.Option(..., help="Path to saved model file"),
    framework: str = typer.Option(..., help="Model framework (sklearn, pytorch, tensorflow)"),
    api: str = typer.Option("fastapi", help="Type of API to generate"),
    output_dir: str = typer.Option("my_app", help="Output directory"),
    name: str = typer.Option(None, "--name", "-n", help="Name for the model (defaults to the filename without extension)"),
    model_class: str = typer.Option(None, "--model-class", help="Path to Python file containing model class definition (required for PyTorch state_dict models)")
):
    """Initialize a new ML model deployment project.
    
    For PyTorch models saved as state_dict, you must provide --model-class pointing to a Python file
    containing the model class definition.
    """
    try:
        # Use the provided name or derive from filename
        if not name:
            name = Path(model).stem
            
        scaffolder = Scaffolder()
        
        # Register the model first
        model_info = scaffolder.register_model(
            name=name,
            version="1.0.0",
            model_path=model,
            framework=framework,
            description=f"Automatically registered by init command on {datetime.now().isoformat()}"
        )
        
        # Then generate the project
        scaffolder.generate_project(
            model_name=name,
            version="1.0.0",
            output_dir=output_dir,
            api_type=api,
            model_class_path=model_class
        )
        
        typer.echo(f"Successfully generated project in {output_dir}")
        typer.echo(f"Model '{name}' v1.0.0 has been registered and is ready to be deployed")
        typer.echo("\nNext steps:")
        typer.echo(f"\t1. cd {output_dir}")
        typer.echo("\t2. docker-compose up --build")
        typer.echo("\n\tYour API will be available at http://localhost:8000")
        typer.echo("\tAPI documentation: http://localhost:8000/docs")
        
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(code=1)

def _print_model_info(model_info: dict):
    """Print detailed information about a model."""
    console.print("\n[bold]Model Information[/bold]")
    console.print(f"  • [bold]Name:[/bold] {model_info['name']}")
    console.print(f"  • [bold]Version:[/bold] {model_info['version']}")
    console.print(f"  • [bold]Framework:[/bold] {model_info['framework']}")
    console.print(f"  • [bold]Path:[/bold] {model_info['path']}")
    console.print(f"  • [bold]Registered:[/bold] {model_info.get('created_at', 'N/A')}")
    
    if 'description' in model_info and model_info['description']:
        console.print("\n[bold]Description:[/bold]")
        console.print(f"  {model_info['description']}")
    
    console.print()  # Add a newline at the end

if __name__ == "__main__":
    app()
