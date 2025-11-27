print("Starting Discord spam detection bot...")

import os
from dotenv import load_dotenv
import discord
from transformers import pipeline, AutoTokenizer
import discord.utils

load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')

print(BOT_TOKEN)

# enable members intent so we can locate moderator members by role.
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

client = discord.Client(intents=intents)


# SETUP MODEL
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
classifier = pipeline('text-classification', model='./spam_transformer_model', tokenizer=tokenizer, top_k=1, device=-1)


# Send DM to this role
MOD_ROLE_NAME = 'Moderator'

async def notify_moderators(guild: discord.Guild, message_obj: discord.Message, prediction_label: str, score: float):
    """
    Send a DM to moderator accounts when a spam message is detected.
    """

    if guild is None:
        # message was a DM or guild not available
        return

    recipients = set()

    # If no explicit IDs, try role lookup
    role = discord.utils.get(guild.roles, name=MOD_ROLE_NAME)
    if role:
        # role.members is cached; if it's empty we try a fetch fallback
        if role.members:
            for m in role.members:
                if not m.bot:
                    recipients.add(m)
        else:
            # fetch all members and filter by role name (API fallback)
            try:
                async for m in guild.fetch_members(limit=None):
                    if any(r.name == MOD_ROLE_NAME for r in m.roles) and not m.bot:
                        recipients.add(m)
            except Exception:
                # If fetching members fails (missing intents/permissions), abort silently
                pass

    # Construct DM content
    jump_url = getattr(message_obj, 'jump_url', 'N/A')
    dm_text = (
        f"[Spam detector] A message was classified as **{prediction_label}** (score={score:.3f}).\n"
        f"Guild: {guild.name} (id={guild.id})\n"
        f"Channel: {message_obj.channel.name if hasattr(message_obj.channel, 'name') else str(message_obj.channel)}\n"
        f"Author: {message_obj.author} (id={message_obj.author.id})\n"
        f"Message: {message_obj.content}\n"
        f"Link: {jump_url}"
    )

    # Send DMs (ignore failures where the moderator has DMs closed)
    for member in recipients:
        try:
            await member.send(dm_text)
        except Exception as e:
            print(f"Failed to DM moderator {member} (id={member.id}): {e}")


@client.event
async def on_ready():
    print(f'We logged in as {client.user}')

@client.event
async def on_message(message: discord.Message):
    # ignore messages from the bot itself
    if message.author == client.user:
        return

    print(f'Seen a message: {message.content}')

    # Run classifier
    try:
        output = classifier(message.content)[0][0] # first message, first output
        label = output.get('label')
        score = float(output.get('score', 0.0))
    except Exception as e:
        print(f'Classification error: {e}')
        return

    result = 'ham' if label == 'LABEL_0' else 'spam'
    print(f'Prediction: {result} (confidence={score:.3f})')

    # If spam, notify moderators
    if result == 'spam':
        await notify_moderators(message.guild, message, result, score)


client.run(BOT_TOKEN)