def api_response(status_code: int = 200, message: str = "", data=None):
    return {"status": status_code, "message": message, "data": data}
