from qiskit import IBMQ
# IBMQ.enable_account('acaf380e8f90520c2f75bae0280ed8f02bd82f7ad1427e88d22793355cf64b606f09118a8f125195019a3e7229042229b5b02ca660995ea2d2a0fcd5bd5ffd7a', 'ibm-q', 'open', 'main')
# IBMQ.load_account() # Load account from disk
# IBMQ.providers()    # List all available providers

# IBMQ.enable_account('1d4191db89ee7da833edc414dd5cf2b08d47a658b81cee8992b14b9c8b49dbcb925945df9a46e5f472612160d8c85762fa78c25fd67ccd5b3aa99d0ca5169d16', hub='ibm-q', group='open', project='main')
# Raz's account: IBMQ.save_account('1d4191db89ee7da833edc414dd5cf2b08d47a658b81cee8992b14b9c8b49dbcb925945df9a46e5f472612160d8c85762fa78c25fd67ccd5b3aa99d0ca5169d16')
# Shira's account: IBMQ.save_account('0bb8d57a4ce7a0f492bfb515c7cab810ff15e014e6cb71263ebc6a8d74809f21f2cf1fe4a9a30ef18e0835a9bb711407da5b799f5fe391d0258f950d610f2783', overwrite='true')
IBMQ.load_account()
# print(IBMQ.providers())

#  IBMQ.delete_accounts()
#  IBMQ.save_account('MY-NEW_TOKEN')
#  IBMQ.load_accounts()



