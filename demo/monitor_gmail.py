import smtplib
import time
import imaplib
import email

FROM_EMAIL  = "sydeitemidentifer@gmail.com"
FROM_PWD    = "SydeFr334all"
SMTP_SERVER = "imap.gmail.com"
SMTP_PORT   = 993

mail = imaplib.IMAP4_SSL(SMTP_SERVER, SMTP_PORT)
mail.login(FROM_EMAIL, FROM_PWD)

def checkMail():
    try:
        mail.select('inbox')
        (retcode, messages) = mail.search(None, '(UNSEEN)')

        if (retcode == 'OK'):
            for num in messages[0].split():
                typ, data = mail.fetch(num, '(RFC822)' )
                for response_part in data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_string(response_part[1].decode('utf-8'))
                        email_subject = msg['subject']
                        email_from = msg['from']
                        if (email_subject == "ItemIdentifier."):
                            print(msg.get_payload().strip())
                            typ, data = mail.store(num,'+FLAGS','\\Seen')
    except Exception as e:
        print(str(e))

while 1:
    print('checking mail')
    checkMail()
    time.sleep(1)
