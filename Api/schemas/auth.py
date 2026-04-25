from pydantic import BaseModel , EmailStr
from datetime import datetime

class UserRegister(BaseModel):

    email : EmailStr
    password : str

class UserLogin(BaseModel):

    email : EmailStr
    password : str

class TokenResponse(BaseModel):

    access_token : str
    token_type : str = "bearer"

class UserOut(BaseModel):

    id : int
    email : EmailStr
    is_active : bool
    is_verified : bool
    created_at : datetime
