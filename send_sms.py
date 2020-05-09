# pip install twilio

from twilio.rest import Client


# put your own credentials here

def sendMessage():
    account_sid = ""
    auth_token = ""
    client = Client(account_sid, auth_token)
    client.messages.create(
        to="",
        from_="",
        body="Stop Touching Your Face!"
    )
