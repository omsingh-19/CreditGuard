from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime, timedelta

from Api.db.session import get_db
from Api.db.models import User
from Api.schemas.auth import UserRegister, UserOut, TokenResponse
from Api.config import settings

from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm


pwd_context = CryptContext(schemes=["bcrypt"])

SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
):
    credentials_error = HTTPException(
        status_code=401,
        detail="Invalid credentials"
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")

        if email is None:
            raise credentials_error

    except JWTError:
        raise credentials_error

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_error

    return user

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=UserOut)
async def user_register(
    user_input: UserRegister,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(User).where(User.email == user_input.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(status_code=400, detail="Email already exists")

    new_user = User(
        email=user_input.email,
        hashed_password=pwd_context.hash(user_input.password),
        is_active=True,
        is_verified=False
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return new_user

@router.post("/login", response_model=TokenResponse)
async def user_login(
    form_data: OAuth2PasswordRequestForm = Depends(),  
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email")

    if not pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    payload = {
        "sub": user.email,
        "exp": datetime.utcnow() + timedelta(minutes=60)
    }

    access_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    return TokenResponse(access_token=access_token)


@router.get("/me", response_model=UserOut)
async def get_me(
    current_user: User = Depends(get_current_user)
):
    return current_user