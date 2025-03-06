"""Database utility functions for executing SQL scripts."""
import logging
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)

async def execute_sql_commands(session: AsyncSession, sql_content: str) -> None:
    """Execute multiple SQL commands from a string, handling dollar-quoted strings properly.

    Args:
        session: The database session
        sql_content: String containing multiple SQL commands
    """
    commands = []
    current_command = []
    in_string = False
    string_char = None
    in_dollar_quote = False
    dollar_quote_tag = None

    i = 0
    while i < len(sql_content):
        char = sql_content[i]

        # Handle dollar quotes (e.g., $BODY$, $function$)
        if char == '$' and not in_string:
            # Check if this is the start of a dollar quote
            if not in_dollar_quote:
                # Look ahead to find the end of the tag
                tag_end = sql_content.find('$', i + 1)
                if tag_end != -1:
                    dollar_quote_tag = sql_content[i:tag_end + 1]
                    in_dollar_quote = True
                    current_command.append(sql_content[i:tag_end + 1])
                    i = tag_end + 1
                    continue
            # Check if this is the end of a dollar quote
            elif dollar_quote_tag:
                if sql_content.startswith(dollar_quote_tag, i):
                    in_dollar_quote = False
                    current_command.append(dollar_quote_tag)
                    i += len(dollar_quote_tag)
                    continue

        # Handle regular string quotes
        if char in ["'", '"'] and not in_dollar_quote:
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None

        # Handle semicolons (command separators)
        if char == ';' and not in_string and not in_dollar_quote:
            current_command.append(char)
            command = ''.join(current_command).strip()
            if command:  # Only add non-empty commands
                commands.append(command)
            current_command = []
        else:
            current_command.append(char)

        i += 1

    # Add the last command if it exists
    last_command = ''.join(current_command).strip()
    if last_command:
        commands.append(last_command)

    # Execute each command in a separate transaction
    for command in commands:
        try:
            logger.debug(f"Executing SQL command:\n{command}")
            await session.execute(text(command))
            await session.commit()
            logger.info("Successfully executed SQL command")
        except Exception as e:
            logger.error(f"Error executing SQL command: {str(e)}\nCommand was:\n{command}")
            await session.rollback()
            raise