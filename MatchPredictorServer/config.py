"""MatchPredictorServer development configuration."""

import pathlib

# Root of this application, useful if it doesn't occupy an entire domain
APPLICATION_ROOT = '/'

# Secret key for encrypting cookies
SECRET_KEY = (b'\xbdB\xa4m\xc8\xbf\xd2zo'
              b'\rm.\x06s\xb8\x88\xd9\xfe\xe9\x84t\x13\x19>')
SESSION_COOKIE_NAME = 'login'

# File Upload to var/uploads/
MATCH_PREDICTOR_ROOT = pathlib.Path(__file__).resolve().parent.parent