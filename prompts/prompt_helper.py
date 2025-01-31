"""
Formatting Prompt templates.
"""

from typing import Dict, Any
from langchain_core.prompts import PromptTemplate

def get_custom_prompt_template(standalone_prompt_template: str, partial_variables: Dict[str, Any] = None) -> PromptTemplate:
    """
    Helper function to create a custom prompt template
    
    Args:
        standalone_prompt_template (str): Standalone prompt to be used.
    
    Returns:
        PromptTemplate: Custom prompt template.
    """    
    # Create a custom prompt template
    return PromptTemplate.from_template(
        template=standalone_prompt_template, partial_variables=partial_variables
    )