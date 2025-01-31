"""Langtrace Helper"""
from typing import Dict
from langfuse.callback import CallbackHandler
from NDISxRAG.settings import LOGGER_NAME, LangfuseConfig
import logging
import uuid
 
logger = logging.getLogger(LOGGER_NAME)
langfuse_handler = CallbackHandler(public_key=LangfuseConfig.PUBLIC_KEY.value, secret_key=LangfuseConfig.SECRET_KEY.value, host=LangfuseConfig.LANGFUSE_HOST.value, trace_name=LangfuseConfig.TRACE_NAME.value, threads=LangfuseConfig.THREADS.value)

def verify_langfuse_connection() -> bool:
    """
    Verifies the connection to the Langfuse service.

    Returns:
        - is_connected (bool): True if the connection is successful, False otherwise.
    """
    try:
        langfuse_handler.auth_check()  # Tests the SDK connection with the server
        return True
    except Exception as e:
        logger.warning("[LangFuse] Langfuse service is unavailable. Please verify your keys and host address. | Error: %s", str(e))
        return False

def get_langfuse_configuration(user_id: int, run_title: str) -> Dict:
    """
    Generates a configuration for running Langfuse with Chains.

    Parameters:
        - message (Message): The message object containing message details (used for Trace ID).
        - run_title (str): The title of the run displayed on the Langfuse dashboard.

    Returns:
        - config (Dict): The configuration details for Langtrace.
    """
    is_service_connected = verify_langfuse_connection()
    logger.info("[LangFuse] Langfuse service connection status: %s", is_service_connected)

    if not is_service_connected:
        return {}

    config = {
        "run_id": str(uuid.uuid4()),
        "run_name": run_title,
        "callbacks": [langfuse_handler],
        "metadata": {
            "user_id": user_id,
        },
        "chat_id": user_id,  # user_id
    }
    logger.info("[LangFuse] Langfuse configuration generated successfully | Run ID: %s | Run Name: %s", config["run_id"], config["run_name"])
    return config
