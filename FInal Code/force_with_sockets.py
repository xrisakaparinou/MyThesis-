import numpy as np
import multiprocessing
from nidaqmx import Task, constants, system
import time
from sympy import symbols, diff, cos, sin, Pow, Abs, sqrt, simplify, evalf
import keyboard
import socket

PPR = 19968
KAPPA = 7
a0 = 5
# a1 = 7.5
a1 = 15.3
a2 = 15.3

host, port = "192.168.1.10", 12345
data = "1,2,3"
force = 0
anglex = 0
angley = 0
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))

queue = multiprocessing.Queue()

def free_tasks():
    t1 = Task('input_task_1')
    t2 = Task('input_task_2')

    t1.stop()
    t1.close()
    t1.__del__()

    t2.stop()
    t2.close()
    t2.__del__()


def signedNum(x, counterNBits):
    signedThreshold = 2 ** (counterNBits - 1)
    if isinstance(x, float):
        signedData = np.array([x])
    else:
        signedData = x[:, 0]
    signedData[signedData > signedThreshold] -= 2 ** counterNBits
    return signedData


def initiate_acquisition_task(task_number):
    task_name = 'input_task_1' if task_number == 1 else 'input_task_2'
    channel = 'Dev1/ctr0' if task_number == 1 else 'Dev1/ctr1'
    source = 'PFI0' if task_number == 1 else 'PFI2'
    #deg = 0 if task_number == 1 else 180
    deg = 60 if task_number == 1 else 120

    acquisition_task = Task(task_name)

    acquisition_task.ci_channels.add_ci_ang_encoder_chan(counter=channel, decoding_type=constants.EncoderType.X_4,
                                                         initial_angle=np.deg2rad(deg),
                                                         units=constants.AngleUnits.TICKS, pulses_per_rev=1024)
    acquisition_task.timing.cfg_samp_clk_timing(source=source, rate=1000, samps_per_chan=10000000,
                                                sample_mode=constants.AcquisitionType.CONTINUOUS)
    print("created new task: ", task_name)

    return acquisition_task


def initiate_output_task(task_number):
    task_name = 'output_task_1' if task_number == 1 else 'output_task_2'
    channel = 'Dev2/ao0' if task_number == 1 else 'Dev2/ao1'

    output_task = Task(task_name)
    output_task.ao_channels.add_ao_voltage_chan(channel, min_val=-10, max_val=10)
    print("created new task: ", task_name)

    return output_task


def calculate_x_y(curr_ang1, curr_ang2):
    ang1, ang2 = symbols('ang1 ang2', real=True)

    xb = a1 * cos(ang2) + a0
    yb = a1 * sin(ang2)
    xd = a1 * cos(ang1)
    yd = a1 * sin(ang1)

    xb = xb.subs({ang2: np.deg2rad(curr_ang2)}).evalf()
    yb = yb.subs({ang2: np.deg2rad(curr_ang2)}).evalf()
    xd = xd.subs({ang1: np.deg2rad(curr_ang1)}).evalf()
    yd = yd.subs({ang1: np.deg2rad(curr_ang1)}).evalf()

    d1 = sqrt((xb - xd) * (xb - xd) + (yb - yd) * (yb - yd))
    # d1 = d1.subs({ang1: np.deg2rad(curr_ang1), ang2: np.deg2rad(curr_ang2)}).evalf()
    # print("d1: ", d1)
    d2 = d1 / 2
    # print("d2: ", d2)
    d3 = sqrt(Abs(a2 ** 2 - d2 ** 2))
    # print("d3: ", d3)

    xp = xb + ((d2 / d1) * (xd - xb))
    yp = yb - ((d2 / d1) * (yb - yd))

    x = xp + (d3 / d1 * (yd - yb))
    y = yp - (d3 / d1 * (xd - xb))

    return x, y


def calculate_t1_t2(curr_ang1, curr_ang2, y_val):
    ang1, ang2 = symbols('ang1 ang2', real=True)
    xb = a1 * cos(ang1) + a0
    yb = a1 * sin(ang1)
    xd = a1 * cos(ang2)
    yd = a1 * sin(ang2)

    d1 = sqrt((xb - xd) * (xb - xd) + (yb - yd) * (yb - yd))
    d2 = d1 / 2
    d3 = sqrt(Abs(a1 ** 2 - d2 ** 2))

    xp = xb + ((d2 / d1) * (xd - xb))
    yp = yb + ((d2 / d1) * (yb - yd))

    x = xp + (d3 / d1 * (yd - yb))
    y = yp - (d3 / d1 * (xd - xb))

    dxg_dang2 = diff(x, ang2)
    dyg_dang2 = diff(y, ang2)
    dyg_dang1 = diff(y, ang1)
    dxg_dang1 = diff(x, ang1)

    t1 = dxg_dang1 * force * sin(anglex) + dyg_dang1 * force * cos(angley)
    t2 = dxg_dang2 * force * sin(anglex) + dyg_dang2 * force * sin(anglex)

    t1_rep = t1.subs({ang1: np.deg2rad(curr_ang1), ang2: np.deg2rad(curr_ang2)})
    t1_val = t1_rep.evalf()
    t2_rep = t2.subs({ang1: np.deg2rad(curr_ang1), ang2: np.deg2rad(curr_ang2)})
    t2_val = t2_rep.evalf()

    return -t1_val, -t2_val


def handle_close(at1, at2, ot1, ot2):
    print("Handling closure of open tasks...")
    at1.stop()
    at1.close()
    at2.stop()
    at2.close()
    ot1.stop()
    ot1.close()
    ot2.stop()
    ot2.close()
    print("Exiting...")
    exit(0)


def check_for_end_condition(queue):
    while True:
        if keyboard.is_pressed("ctrl + q"):
            print("Stopping acquisition due to keyboard interrupt")
            queue.put(True)
            break


def main():

    global data
    global force, anglex, angley
    x,y = 0.00, 0.00

    p1 = multiprocessing.Process(target=check_for_end_condition, args=(queue,))
    p1.start()

    try:
        print("Start")
        print('getting rid of existing tasks')
        free_tasks()

        acquisition_task_1 = initiate_acquisition_task(1)
        acquisition_task_2 = initiate_acquisition_task(2)

        output_task_1 = initiate_output_task(1)
        output_task_2 = initiate_output_task(2)

        # previous_ang1 = 180
        # previous_ang2 = 0
        previous_ang1 = 120
        previous_ang2 = 60

        while queue.empty():

            try:
                read_value_2 = acquisition_task_1.read(number_of_samples_per_channel=20, timeout=0)[0]
                # print("rv1", read_value_1)
                position2 = signedNum(read_value_2, 32)
                degrees2 = (position2 / PPR) * 180
                # curr_ang2 = -degrees2[0]
                curr_ang2 = -degrees2[0] + 60
                # print("degrees2",degrees2)

                # print("rv2: ", read_value_1)

            except Exception as e:
                # print(e)
                curr_ang2 = previous_ang2
                pass

            try:
                read_value_1 = acquisition_task_2.read(number_of_samples_per_channel=20, timeout=0)[0]
                # print("rv2:", read_value_2)
                position1 = signedNum(read_value_1, 32)
                degrees1 = (position1 / PPR) * 180
                #curr_ang1 = -degrees1[0] + 180
                curr_ang1 = -degrees1[0] + 160
                # print("degrees1", degrees1)
                # print("rv1: ", read_value_2)


            except Exception as e:
                # print(e)
                curr_ang1 = previous_ang1
                pass

            x, y = calculate_x_y(curr_ang1, curr_ang2)

            x1 = round(x/10, 3)
            y1 = round(y/10, 3)
            data = str(x1) + "," + str(y1) + ",-0.54"
            sock.sendall(data.encode("utf-8"))
            res = sock.recv(1024).decode("utf-8")


            str_force, str_anglex, str_angley = res.split(",")


            force = float(str_force)
            anglex = float(str_anglex)
            angley = float(str_angley)

            write_value_1 = 0
            write_value_2 = 0
            if force > 0:

                t1, t2 = calculate_t1_t2(curr_ang1, curr_ang2, y)
                max_t = 1.096  # Ν*cm
                # print("t1", t1, " ", "t2", t2)
                #kinitiras deksia
                if t1 < -max_t:
                    write_value_1 = 10
                elif t1 > max_t:
                    write_value_1 = 10
                else:
                    print("t1:",t1)
                    t_ratio = t1 / max_t

                    if t1 > 0:
                        write_value_1 = t_ratio * 10
                    else:
                        write_value_1 = -t_ratio * 10

                if t2 < -max_t:
                    print("w2max: case1")
                    write_value_2 = -10
                elif t2 > max_t:
                    print("w2max: case2")
                    write_value_2 = -10
                else:
                    print("w2nonmax")
                    print("t2:", t2)
                    t_ratio = t2 / max_t

                    if t2 > 0:
                        print("w2nonmax: case 1")
                        write_value_2 = -t_ratio * 10
                    else:
                        print("w2nonmax: case 2")
                        write_value_2 = t_ratio * 10
            # print("wr1:", write_value_1)
            # print("wr2:", write_value_2)
            output_task_1.write(write_value_1, timeout=0)
            output_task_2.write(write_value_2, timeout=0)


            if previous_ang1 != curr_ang1 or previous_ang2 != curr_ang2:
                print("ang1: ", curr_ang1)
                print("ang2: ", curr_ang2)
                print("x: ", x)
                print("y: ", y)

            previous_ang1 = curr_ang1
            previous_ang2 = curr_ang2

        print("Exiting loop.")
        print("Stopping and closing active tasks.")
        handle_close(acquisition_task_1, acquisition_task_2, output_task_1, output_task_2)

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
