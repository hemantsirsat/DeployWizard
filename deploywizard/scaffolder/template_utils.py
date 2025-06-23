from typing import Dict, Any

def get_template_vars(model_path: str, framework: str) -> Dict[str, Any]:
    """Get template variables based on model and framework."""
    return {
        'model_path': model_path,
        'framework': framework,
        'dependencies': {
            'fastapi': '0.104.1',
            'uvicorn': '0.24.0',
            'pydantic': '2.4.2',
            'scikit-learn': '1.3.0' if framework == 'sklearn' else None,
            'torch': '2.1.0' if framework == 'pytorch' else None,
            'tensorflow': '2.13.0' if framework == 'tensorflow' else None
        }
    }
