import urllib.request
import urllib.parse
from src.config import cfg

def notify(title: str, message: str, priority: str='3') -> bool:
    try:
        data = message.encode('utf-8')
        req = urllib.request.Request(f'https://ntfy.sh/{cfg.ntfy_topic}', data=data, headers={'Title': urllib.parse.quote(title, safe=''), 'Priority': str(priority)})
        urllib.request.urlopen(req, timeout=5)
        print(f'[ntfy] Sent: {title}')
        return True
    except Exception as e:
        print(f'[ntfy] FAILED ({type(e).__name__}): {e}')
        return False
