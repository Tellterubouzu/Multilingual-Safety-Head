import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os
def send_email(email_address="shimomura.teruki174@mail.kyutech.jp", password="4bxRLtu2", to_address="shimomura.teruki174@mail.kyutech.jp", subject = "Notify from University Of Tokyo Wisteria Server", body= "完了通知",attachment_paths = None):
    # OutlookのSMTPサーバー情報
    smtp_server = 'smtp-mail.outlook.com'
    smtp_port = 587

    # メール作成
    msg = MIMEMultipart()
    msg['From'] = email_address
    msg['To'] = to_address
    msg['Subject'] = subject

    # メール本文を追加
    msg.attach(MIMEText(body, 'plain'))

    if attachment_paths:
        for file_path in attachment_paths:
            try:
                with open(file_path,'rb') as f:
                    file_data = f.read()
                    file_name = os.path.basename(file_path)
                    part  =MIMEApplication(file_data,Name = file_name)
                    part['Content-Disposition'] = f'attachment; filename="{file_name}"'
                    msg.attach(part)
            except Exception as e:
                print(f"ファイルの添付に失敗しました {file_path}: {e}")


    try:
        # SMTPサーバーへ接続
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # TLS暗号化を開始
        # ログイン
        server.login(email_address, password)
        # メール送信
        server.sendmail(email_address, to_address, msg.as_string())
        # サーバーとの接続を終了
        server.quit()
        #print(f"{subject}")
        #print("Notify Mail send Normaly Completed")
    except Exception as e:
        print(f"{subject}")
        print(f"{e} has Occured, the Main cound'nt send normally")

# 使い方
# send_email('your_outlook_email@example.com', 'your_password', 'recipient@example.com', 'メールのタイトル', 'これはテストメールです。')