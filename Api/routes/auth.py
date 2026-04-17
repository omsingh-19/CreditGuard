from fastapi import APIRouter , Depends
from Api.db.session import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from Api.db.models import User
from datetime import datetime , timedelta
from sqlalchemy.future import select
from fastapi.exceptions import HTTPException
from Api.schemas.auth import UserRegister , UserOut  , UserLogin , TokenResponse
from passlib.context import CryptContext
from jose import jwt , JWTError
from fastapi.security import OAuth2PasswordBearer
from Api.config import settings


pwd_context = CryptContext(schemes=["bcrypt"])

SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

async def get_current_user(
        token : str = Depends(oauth2_scheme),
        db : AsyncSession = Depends(get_db)
):
    credential_error = HTTPException(401 , detail="Invalid Credentials")

    try: 
        payload = jwt.decode(token,SECRET_KEY,algorithms=[ALGORITHM])
        email : str = payload.get("sub")

        if not email:
            raise credential_error
        
    except JWTError:
        raise credential_error
    
    result =await db.execute(select(User).where(User.email==email))
    user = result.scalar_one_or_none()

    if not user:
        raise credential_error
    
    return user



router = APIRouter(prefix="/auth",tags=["auth"])
@router.post("/register",response_model=UserOut)
async def user_register(
    user_input : UserRegister ,
    db : AsyncSession = Depends(get_db)
):
    result = await db.execute(select(User).where(User.email == user_input.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(400 , detail= "Email already exists")

    new_user = User(
        email = user_input.email,
        hashed_password = pwd_context.hash(user_input.password),
        is_active = True,
        is_verified = False)

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return new_user

@router.post("/login",response_model=TokenResponse)
async def user_login(
    user_login_info : UserLogin,
    db : AsyncSession = Depends(get_db)
):
    result = await db.execute(select(User).where(User.email == user_login_info.email))
    existing_user_info = result.scalar_one_or_none()

    if not existing_user_info:
        raise HTTPException(401 , detail= "Incorrect Email")
    
    if not pwd_context.verify(user_login_info.password,existing_user_info.hashed_password):
        raise HTTPException(401 , "Invalid Credentials")
    
    payload = {
        "sub": user_login_info.email,
        "exp": datetime.utcnow() + timedelta(minutes=60)
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    return TokenResponse(access_token=token)
    
@router.get("/me",response_model=UserOut)
async def get_me(
    current_user : User = Depends(get_current_user)
):
    return current_user