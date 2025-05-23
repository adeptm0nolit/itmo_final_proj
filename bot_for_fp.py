
import asyncio
from token import AWAIT

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, BotCommand, BufferedInputFile
from aiogram.methods.delete_message import DeleteMessage
from aiogram.utils.keyboard import InlineKeyboardBuilder
from pathlib import Path
from aiogram.fsm.state import State, StatesGroup
import numpy as np
import matplotlib.pyplot as plt
import io

from pyexpat.errors import messages

from vae_1_fp import gen_num, decoder, vect


API_TOKEN = Path("C:\\Users\\pimen\\OneDrive\\Decktop\\api.txt").read_text()

dp = Dispatcher()

bot = Bot(token=API_TOKEN)

class Form(StatesGroup):
    number = State()
    coefficients = State()
    power = State()
    x_range = State()
    func_type = State()

async def main():
  await dp.start_polling(bot)


async def set_bot_commands():
    commands = [
        BotCommand(command="/start", description="Start the bot"),
        BotCommand(command="/generate", description="Generate a number image"),
        BotCommand(command="/plot", description="Create a function plot")
    ]
    await bot.set_my_commands(commands)


# Обработчик для команды /start
@dp.message(Command('start'))
async def start(msg: Message):
    await msg.answer(f'Bot started successfully!')


# Обработчик для команды /generate
@dp.message(Command('generate'))
async def get_usr_num(message: Message, state: FSMContext):
    await state.set_state(Form.number)
    await message.answer("Enter a number, which you want to generate")


#Считывание ввода пользователя и проверка, что введено число
@dp.message(Form.number)
async def get_usr_num(message: Message, state: FSMContext):
    if not message.text.isdigit():
        await message.answer("It must be a number!")
        return

    await state.update_data(number=message.text)
    data = await state.get_data()
    num = data['number']

    bot_msg = await message.answer('The image is being generated')

    await gen_img(num, message.chat.id, bot_msg.message_id)


#Генерация картинки с числом
async def gen_img(num: str, chat_id: int, msg_id: int):
    num_len = len(num)
    # Раскодируем (восстановим) изображения для выбранных точек
    reconstructions = gen_num(num, vect, decoder)

    # Выведем получившиеся изображения и их коды
    fig = plt.figure()
    fig.subplots_adjust(hspace=0, wspace=-0.15)
    img_buf = io.BytesIO()

    for i in range(num_len - 1, -1, -1):
        ax = fig.add_subplot(1, num_len, i + 1)
        ax.axis("off")
        ax.imshow(reconstructions[i, :, :], cmap="gray")

    plt.savefig(img_buf, bbox_inches='tight', pad_inches=0)
    img_buf.seek(0)
    plt.close()

    image = BufferedInputFile(img_buf.read(), filename="plot.png")

    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(text="Yes", callback_data='regenerate'))
    builder.add(types.InlineKeyboardButton(text="No", callback_data='skip'))
    await bot(DeleteMessage(chat_id=chat_id, message_id=msg_id))
    await bot.send_photo(chat_id=chat_id, photo=image,
                         caption='Here`s your number\n\nDo you want to generate number again?',
                         reply_markup=builder.as_markup())


#Обработчик повторной генерации
@dp.callback_query(Form.number)
async def process_regen(callback: types.CallbackQuery, state: FSMContext):
    if callback.data == 'regenerate':
        data = await state.get_data()
        num = data['number']
        bot_msg = await callback.message.answer('The image is being generated')
        await gen_img(num, callback.message.chat.id, bot_msg.message_id)
    else:
        await state.clear()
        await callback.message.answer("Okay, generation stopped. \nUse /generate to start again or /plot to create a function plot.")

    await callback.answer()

# Обработчик для команды /plot
@dp.message(Command('plot'))
async def get_func_type(message: Message):
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(text="y = kx + b", callback_data='linear'))
    builder.add(types.InlineKeyboardButton(text="y = a1 * x ^ n + a2 * x ^ (n-1) ...", callback_data='polynomial'))
    builder.add(types.InlineKeyboardButton(text="y = k/(x ^ n + b)", callback_data='rational'))
    builder.add(types.InlineKeyboardButton(text="Trigonometry functions", callback_data='trigonometry'))
    builder.adjust(1)
    await message.answer("Select plot of which function type you want to create:", reply_markup=builder.as_markup())

#Обработчик выбора типа функции
@dp.callback_query()
async def get_f_type(callback: types.CallbackQuery, state: FSMContext):
    if callback.data in ['linear', 'polynomial', 'rational', 'trigonometry']:
        if callback.data == 'linear':
            await state.set_state(Form.coefficients)
            await state.update_data(func_type='linear')
            await callback.message.answer("Enter coefficients (k and b)")
        elif callback.data == 'polynomial':
            await state.set_state(Form.power)
            await state.update_data(func_type='polynomial')
            await callback.message.answer("Enter power of polynomial")
        elif callback.data == 'rational':
            await state.set_state(Form.power)
            await state.update_data(func_type='rational')
            await callback.message.answer("Enter power of x")
        elif callback.data == 'trigonometry':
            builder = InlineKeyboardBuilder()
            await state.update_data(func_type='trigonometry')
            builder.add(types.InlineKeyboardButton(text="y = sin(kx + b)", callback_data='sin'))
            builder.add(types.InlineKeyboardButton(text="y = tan(kx + b)", callback_data='tan'))
            builder.add(types.InlineKeyboardButton(text="y = sinh(kx + b)", callback_data='sinh'))
            builder.add(types.InlineKeyboardButton(text="y = tanh(kx + b)", callback_data='tanh'))
            builder.adjust(1)
            await callback.message.answer("Select type of trigonometric function:", reply_markup=builder.as_markup())
    else:
        if callback.data in ['sin', 'tan', 'sinh', 'tanh']:
            await state.update_data(func_type=callback.data)
            await state.set_state(Form.coefficients)
            await callback.message.answer("Enter coefficients (k and b)")

    await callback.answer()

#Проверка введенной степени
@dp.message(Form.power)
async def get_power(message: Message, state: FSMContext):
    if not message.text.isdigit():
        await message.answer("Power must be a number!")
        return

    await state.update_data(power=int(message.text))
    data = await state.get_data()

    await state.set_state(Form.coefficients)

    if data['func_type'] == 'polynomial':
        await message.answer(f"Enter coefficients from a1 to a{int(message.text) + 1}")
    else:
        await message.answer("Enter coefficients (k and b)")

#Проверка введённых коэффициентов
@dp.message(Form.coefficients)
async def get_coefficients(message: Message, state: FSMContext):
    t = list(message.text.split(' '))
    for i in t:
        if not i.isdigit() and not (i[0] == '-' and i[1:].isdigit()):
            await message.answer("Coefficient must be a number!")
            return
    t = list(map(int, t))
    await state.update_data(coefficients=t)

    await state.set_state(Form.x_range)
    await message.answer("Enter a range of x")

#Проверка введенного промежутка для X
@dp.message(Form.x_range)
async def get_x_range(message: Message, state: FSMContext):
    t = list(message.text.split(' '))
    if len(t) != 2:
        await message.answer("Range of x must be 2 numbers (min and max)")
        return
    for i in t:
        if not i.isdigit() and not (i[0] == '-' and i[1:].isdigit()):
            await message.answer("x must be a number!")
            return
    t = list(map(int, t))
    await state.update_data(x_range=t)

    bot_msg = await message.answer('The plot is being created')
    msg_id = bot_msg.message_id

    data = await state.get_data()
    f_type = data['func_type']
    chat_id = message.chat.id

    if f_type == 'linear':
        await linear_func(data, chat_id, msg_id)
    elif f_type == 'polynomial':
        await polynomial_func(data, chat_id, msg_id)
    elif f_type == 'rational':
        await rational_func(data, chat_id, msg_id)
    else:
        await trigon_func(data, chat_id, msg_id)

#Создание массивов для построения графиков
#Линейная функция
async def linear_func(data: dict, chat_id: int, msg_id: int):
    x_min, x_max = float(min(data['x_range'])), float(max(data['x_range']))
    arg = np.array([i for i in np.linspace(x_min, x_max, 1000)])
    k, b = data['coefficients'][0], data['coefficients'][1]
    y = arg * k
    y = y + b
    await create_plot(arg, y, chat_id, msg_id)

#Многочлен n-ной степени
async def polynomial_func(data: dict, chat_id: int, msg_id: int):
    x_min, x_max = float(min(data['x_range'])), float(max(data['x_range']))
    arg = np.array([i for i in np.linspace(x_min, x_max, 1000)])
    power, coef = data['power'], data['coefficients']
    y = np.array([0 for _ in range(1000)])
    for i in range(power, -1, -1):
        y = y + (np.power(arg, i) * coef[power - i])
    await create_plot(arg, y, chat_id, msg_id)

#Рациональная функция
async def rational_func(data: dict, chat_id: int, msg_id: int):
    x_min, x_max = float(min(data['x_range'])), float(max(data['x_range']))
    arg = np.array([i for i in np.linspace(x_min, x_max, 1000)])
    power, k, b = data['power'], data['coefficients'][0], data['coefficients'][1]
    y = k / (np.power(arg, power) + b)
    await create_plot(arg, y, chat_id, msg_id, lim=True)

#Тригонометрические функции
async def trigon_func(data: dict, chat_id: int, msg_id: int):
    f_type = data['func_type']
    x_min, x_max = float(min(data['x_range'])), float(max(data['x_range']))
    arg = np.array([i for i in np.linspace(x_min, x_max, 1000)])
    k, b = data['coefficients'][0], data['coefficients'][1]
    if f_type == 'sin':
        y = np.sin(arg * k + b)
    elif f_type == 'tan':
        y = np.tan(arg * k + b)
        await create_plot(arg, y, chat_id, msg_id, tan = True)
        lim = True
    elif f_type == 'sinh':
        y = np.sinh(arg * k + b)
    else:
        y = np.tanh(arg * k + b)
    await create_plot(arg, y, chat_id, msg_id)


#Построение и отправка готового графика
async def create_plot(x, y: np.array, chat_id: int, msg_id: int, lim: bool = False, tan: bool = False):
    img_buf = io.BytesIO()

    plt.xlabel('X')
    plt.ylabel('Y')
    if lim:
        plt.ylim(top=max(x) + 10)
        plt.ylim(bottom=min(x) - 10)
    if tan:
        plt.ylim(top=10)
        plt.ylim(bottom=-10)
    plt.grid()
    plt.plot(x, y)

    plt.savefig(img_buf, pad_inches=1, dpi=300)
    img_buf.seek(0)
    plt.close()
    image = BufferedInputFile(img_buf.read(), filename="plot.png")
    await bot(DeleteMessage(chat_id=chat_id, message_id=msg_id))
    await bot.send_photo(chat_id=chat_id, photo=image, caption='Here`s your plot\nUse /generate to start again or /plot to create a function plot.')


if __name__ == '__main__':
    asyncio.run(main())