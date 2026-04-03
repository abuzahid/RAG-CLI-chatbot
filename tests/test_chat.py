import pytest
from src.chat.session import ChatSession
from src.config import reset_config


@pytest.fixture(autouse=True)
def reset_config_fixture():
    """Reset config between tests"""
    reset_config()
    yield
    reset_config()


def test_session_initialization():
    """Test that session initializes with empty history"""
    session = ChatSession(max_history=5)
    assert len(session.history) == 0


def test_session_adds_messages():
    """Test adding messages to session"""
    session = ChatSession(max_history=5)
    session.add_user_message("Hello")
    session.add_ai_message("Hi there!")

    assert len(session.history) == 2
    assert session.history[0]["role"] == "user"
    assert session.history[0]["content"] == "Hello"


def test_session_respects_max_history():
    """Test that session respects max history limit"""
    session = ChatSession(max_history=3)

    session.add_user_message("1")
    session.add_ai_message("1")
    session.add_user_message("2")
    session.add_ai_message("2")
    session.add_user_message("3")
    session.add_ai_message("3")

    # Should only keep last 3 messages
    assert len(session.history) == 3
    assert session.history[0]["content"] == "2"


def test_session_clear():
    """Test clearing session history"""
    session = ChatSession(max_history=5)
    session.add_user_message("Test")

    session.clear()

    assert len(session.history) == 0


def test_session_get_history_string():
    """Test getting formatted history string"""
    session = ChatSession(max_history=5)
    session.add_user_message("What is life insurance?")
    session.add_ai_message("Life insurance is...")

    history = session.get_history_string()

    assert "What is life insurance?" in history
    assert "Life insurance is..." in history


def test_session_from_config():
    """Test creating session from config"""
    session = ChatSession.from_config()

    assert session.max_history == 10  # From config.yaml
