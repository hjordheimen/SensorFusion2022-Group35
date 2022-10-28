from dataclasses import dataclass, field
from tkinter import N
from typing import Tuple
import numpy as np
import scipy.linalg
from senfuslib import MultiVarGauss, DynamicModel
from states import (ErrorState, ImuMeasurement,
                    CorrectedImuMeasurement, NominalState,
                    GnssMeasurement, EskfState)
from quaternion import RotationQuaterion
from utils.indexing import block_3x3
from utils.cross_matrix import get_cross_matrix
from solution import models as models_solu


@dataclass
class ModelIMU:
    """The IMU is considered a dynamic model instead of a sensar. 
    This works as an IMU measures the change between two states, 
    and not the state itself.."""

    accm_std        : float     
    accm_bias_std   : float     #Bias noise in accelerometer, a_w
    accm_bias_p     : float     #Noise on accelerometer, a_n???

    gyro_std        : float
    gyro_bias_std   : float     #Bias noise in gyro, w_w
    gyro_bias_p     : float     #Noise on gyro, w_n???

    accm_correction : 'np.ndarray[3, 3]'
    gyro_correction : 'np.ndarray[3, 3]'

    g: 'np.ndarray[3]'          = field(default=np.array([0, 0, 9.82]))

    Q_c: 'np.ndarray[12, 12]'   = field(init=False, repr=False)

    def __post_init__(self):
        def diag3(x): return np.diag([x]*3)

        accm_corr = self.accm_correction
        gyro_corr = self.gyro_correction

        self.Q_c = scipy.linalg.block_diag(
            accm_corr @ diag3(self.accm_std**2) @ accm_corr.T,
            gyro_corr @ diag3(self.gyro_std**2) @ gyro_corr.T,
            diag3(self.accm_bias_std**2),
            diag3(self.gyro_bias_std**2)
        )

    def correct_z_imu(self,
                      x_est_nom: NominalState,
                      z_imu: ImuMeasurement,
                      ) -> CorrectedImuMeasurement:
        """Correct IMU measurement so it gives a measurmenet of acceleration 
        and angular velocity in body.

        Hint: self.accm_correction and self.gyro_correction translates 
        measurements from IMU frame (probably not correct name) to body frame

        Args:
            x_est_nom: previous nominal state
            z_imu: raw IMU measurement

        Returns:
            z_corr: corrected IMU measurement
        """
        a_b, w_b    = x_est_nom.accm_bias, x_est_nom.gyro_bias
        a_m, w_m    = z_imu.acc, z_imu.avel

        # Check here if failure later...
        acc_est     = self.accm_correction @ (a_m - a_b)
        avel_est    = self.gyro_correction @ (w_m - w_b)

        z_corr      = CorrectedImuMeasurement(acc_est, avel_est)
        
        return z_corr

    def predict_nom(self,
                    x_est_nom: NominalState,
                    z_corr: CorrectedImuMeasurement,
                    dt: float) -> NominalState:
        """Predict the nominal state, given a corrected IMU measurement and a 
        time step, by discretizing (10.58) in the book.

        We assume the change in orientation is negligable when caculating 
        predicted position and velicity, see assignment pdf.

        Hint: You can use: delta_rot = RotationQuaterion.from_avec(something)

        Args:
            x_est_nom: previous nominal state
            z_corr: corrected IMU measuremnt
            dt: time step
        Returns:
            x_nom_pred: predicted nominal state
        """
        pos, vel, ori   = x_est_nom.pos, x_est_nom.vel, x_est_nom.ori
        acc_b, gyro_b   = x_est_nom.accm_bias, x_est_nom.gyro_bias

        a, w            = RotationQuaterion.as_rotmat(ori) @ z_corr.acc + self.g, z_corr.avel

        # print("_______________________________")
        # print(self.accm_bias_p, self.accm_bias_std)
        # print("_______________________________")
        
        

        pos_pred        = pos + dt * vel + dt**2 * a / 2  # TODO
        vel_pred        = vel + dt * a  # TODO

        delta_rot       = RotationQuaterion.from_avec(dt*w)  # TODO
        ori_pred        = RotationQuaterion.multiply(ori, delta_rot)  # TODO

        acc_bias_pred   = acc_b - self.accm_bias_p * np.eye(3) @ acc_b *dt # TODO
        gyro_bias_pred  = gyro_b - self.gyro_bias_p * np.eye(3) @ gyro_b * dt  # TODO

        x_nom_pred      = NominalState(pos_pred, vel_pred, ori_pred, acc_bias_pred, gyro_bias_pred)
        
        return x_nom_pred

    def A_c(self,
            x_est_nom: NominalState,
            z_corr: CorrectedImuMeasurement,
            ) -> 'np.ndarray[15, 15]':
        """Get the transition matrix, A, in (10.68)

        Hint: The S matrices can be created using get_cross_matrix. In the book
        a perfect IMU is expected (thus many I matrices). Here we have 
        to use the correction matrices, self.accm_correction and 
        self.gyro_correction, instead of som of the I matrices.  

        You can use block_3x3 to simplify indexing if you want to.
        ex: first I element in A can be set as A[block_3x3(0, 1)] = np.eye(3)

        Args:
            x_nom_prev: previous nominal state
            z_corr: corrected IMU measurement
        Returns:
            A (ndarray[15,15]): A
        """
        A_c = np.zeros((15, 15))
        Rq = x_est_nom.ori.as_rotmat()
        S_acc = get_cross_matrix(z_corr.acc)
        S_omega = get_cross_matrix(z_corr.avel)

        A_c[block_3x3(0, 1)]    = np.eye(3)
        A_c[block_3x3(1, 2)]    = -Rq @ S_acc
        A_c[block_3x3(1, 3)]    = -Rq @ self.accm_correction
        A_c[block_3x3(2, 2)]    = -S_omega
        A_c[block_3x3(2, 4)]    = -self.gyro_correction
        A_c[block_3x3(3, 3)]    = -self.accm_bias_p * np.eye(3)
        A_c[block_3x3(4, 4)]    = -self.gyro_bias_p * np.eye(3)

        # TODO remove this
        # A_c = models_solu.ModelIMU.A_c(self, x_est_nom, z_corr)

        return A_c

    def get_error_G_c(self,
                      x_est_nom: NominalState,
                      ) -> 'np.ndarray[15, 15]': # Should be [15, 12]
        """The continous noise covariance matrix, G, in (10.68)

        Hint: you can use block_3x3 to simplify indexing if you want to.
        The first I element in G can be set as G[block_3x3(2, 1)] = -np.eye(3)

        Args:
            x_est_nom: previous nominal state
        Returns:
            G_c (ndarray[15, 15]): G in (10.68) # Should be [15, 12]
        """
        G_c                     = np.zeros((15, 12))
        Rq                      = x_est_nom.ori.as_rotmat()

        G_c[block_3x3(1, 0)]    = -Rq
        G_c[block_3x3(2, 1)]    = -np.eye(3)
        G_c[block_3x3(3, 2)]    = np.eye(3)
        G_c[block_3x3(4, 3)]    = np.eye(3)

        return G_c

    def get_discrete_error_diff(self,
                                x_est_nom: NominalState,
                                z_corr: CorrectedImuMeasurement,
                                dt: float
                                ) -> Tuple['np.ndarray[15, 15]',
                                           'np.ndarray[15, 15]']:
        """Get the discrete equivalents of A and GQGT in (4.63)

        Hint: Use scipy.linalg.expm to get the matrix exponential

        See (4.5 Discretization) and (4.63) for more information. 
        Or see "Discretization of process noise" in 
        https://en.wikipedia.org/wiki/Discretization

        Args:
            x_est_nom: previous nominal state
            z_corr: corrected IMU measurement
            dt: time step
        Returns:
            A_d (ndarray[15, 15]): discrede transition matrix
            GQGT_d (ndarray[15, 15]): discrete noise covariance matrix
        """
        A_c             = self.A_c(x_est_nom, z_corr)  # TODO
        G_c             = self.get_error_G_c(x_est_nom)  # TODO
        GQGT_c          = G_c @ self.Q_c @ G_c.T  # TODO

        exponent        = np.vstack([np.hstack([-A_c, GQGT_c]), \
                                        np.hstack([np.zeros([15, 15]), A_c.T])]) # TODO
        VanLoanMatrix   = scipy.linalg.expm(exponent * dt)  # TODO

        A_d             = VanLoanMatrix[15:30, 15:30].T  # TODO
        GQGT_d          = VanLoanMatrix[0:15, 15:30]  # TODO

        return A_d, GQGT_d

    def predict_err(self,
                    x_est_prev: EskfState,
                    z_corr: CorrectedImuMeasurement,
                    dt: float,
                    ) -> MultiVarGauss[ErrorState]:
        """Predict the error state

        Hint: This is doing a discrete step of (10.68) where x_err 
        is a multivariate gaussian.

        Args:
            x_est_prev: previous estimated eskf state
            z_corr: corrected IMU measuremnt
            dt: time step
        Returns:
            x_err_pred: predicted error state gaussian
        """
        x_est_prev_nom  = x_est_prev.nom     #NominalState
        x_est_prev_err  = x_est_prev.err     #MultivarGauss
        
        Ad, GQGTd       = self.get_discrete_error_diff(x_est_prev_nom, z_corr, dt)  # TODO     

        Qd              = Ad @ GQGTd

        x_err           = Ad @ x_est_prev_err.mean

        P_pred          = Ad @ x_est_prev_err.cov @ Ad.T + Qd

        x_err_pred      = MultiVarGauss(x_err, P_pred)
        
        return x_err_pred
