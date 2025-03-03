import string
import secrets

def create_random_key(length: int = 25) -> str:
    """Generate a random string of specified length."""
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))
